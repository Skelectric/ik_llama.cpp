#include "llama-engram.h"

#include <random>

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "llama-arch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

// ggml Q8_0 block: one FP16 scale + 32 int8 values, tightly packed (34 bytes).
namespace {

struct engram_q8_0_block {
    uint16_t d;      // ggml_fp16_t scale
    int8_t   qs[32];
};

static_assert(sizeof(engram_q8_0_block) == 34, "Q8_0 block must be tightly packed");

// little-endian host assumed: the sidecar is written LE by engram_constants.py
bool read_exact(FILE * f, void * dst, size_t len) {
    return fread(dst, 1, len, f) == len;
}

bool read_u32(FILE * f, uint32_t & v) {
    return read_exact(f, &v, sizeof(v));
}

bool read_i64(FILE * f, int64_t & v) {
    return read_exact(f, &v, sizeof(v));
}

} // namespace

bool llama_engram_constants::load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    bool ok = true;

    char magic[8] = {};
    ok = ok && read_exact(f, magic, sizeof(magic));
    ok = ok && memcmp(magic, "V41EGRM1", 8) == 0;

    uint32_t n_layers = 0;
    ok = ok && read_u32(f, n_layers);
    ok = ok && n_layers > 0 && n_layers <= 64;

    if (ok) {
        layers.resize(n_layers);
        for (auto & l : layers) {
            ok = ok && read_u32(f, l.layer_id);
            for (uint32_t g = 0; g < N_GRAM; ++g) {
                ok = ok && read_i64(f, l.mults[g]);
            }
            for (uint32_t k = 0; k < N_COLS; ++k) {
                ok = ok && read_u32(f, l.primes[k]);
            }

            // offsets: N_COLS + 1 entries, a cumsum starting at 0; the last entry is
            // the layer's total row count (sum of the primes)
            uint32_t offs[N_COLS + 1] = {};
            for (uint32_t k = 0; k <= N_COLS; ++k) {
                ok = ok && read_u32(f, offs[k]);
            }

            ok = ok && offs[0] == 0;
            for (uint32_t k = 0; ok && k < N_COLS; ++k) {
                l.offsets[k] = offs[k];
                ok = offs[k + 1] == offs[k] + l.primes[k]; // contiguous bucket ranges
            }
            if (ok) {
                l.rows = offs[N_COLS];
            }
        }
    }

    uint32_t pad = 0, cvocab = 0, nvocab = 0;
    ok = ok && read_u32(f, pad);
    ok = ok && read_u32(f, cvocab);
    ok = ok && read_u32(f, nvocab);
    ok = ok && cvocab > 0 && nvocab > 0 && pad < cvocab;

    if (ok) {
        pad_compressed   = (int32_t) pad;
        compressed_vocab = cvocab;
        n_vocab          = nvocab;

        token_map.resize(nvocab);
        ok = ok && read_exact(f, token_map.data(), (size_t) nvocab * sizeof(int32_t));
        for (uint32_t i = 0; ok && i < nvocab; ++i) {
            ok = token_map[i] >= 0 && (uint32_t) token_map[i] < cvocab;
        }
    }

    // multipliers: odd, and bounded so id*mult cannot overflow (reference bound);
    // primes: at least 2
    for (const auto & l : layers) {
        for (uint32_t g = 0; ok && g < N_GRAM; ++g) {
            ok = l.mults[g] > 0 && (l.mults[g] & 1) == 1 &&
                 (uint64_t) l.mults[g] <= (uint64_t) INT64_MAX / (cvocab > 0 ? cvocab : 1);
        }
        for (uint32_t k = 0; ok && k < N_COLS; ++k) {
            ok = l.primes[k] >= 2;
        }
    }

    fclose(f);

    if (!ok) {
        layers.clear();
        token_map.clear();
        pad_compressed   = 0;
        compressed_vocab = 0;
        n_vocab          = 0;
    }

    return ok;
}

void llama_engram_hash_row(const llama_engram_constants & c, uint32_t e, const int32_t * ids, uint32_t * rows) {
    const auto & l = c.layers[e];

    // all operands are non-negative, so the rolling XOR stays non-negative and the
    // uint64 modulo matches the reference's Python-semantics int64 modulo
    uint64_t hash = (uint64_t) ids[0] * (uint64_t) l.mults[0];

    for (uint32_t g = 1; g < c.N_GRAM; ++g) {
        hash ^= (uint64_t) ids[g] * (uint64_t) l.mults[g];

        // the running value after step g is the hash of the (g+1)-gram; each head
        // reduces it modulo its own prime, landing in a disjoint bucket range
        for (uint32_t h = 0; h < c.N_HEAD; ++h) {
            const uint32_t col = (g - 1) * c.N_HEAD + h;

            rows[col] = (uint32_t) (hash % l.primes[col]) + l.offsets[col];
        }
    }
}

void llama_engram_dequant_row_q8_0(const uint8_t * row, float * dst) {
    // rows are 272-byte strided, so blocks are 2-byte aligned (uint16_t reads are fine)
    const auto * blk = (const engram_q8_0_block *) (const void *) row;

    for (uint32_t b = 0; b < llama_engram_constants::HEAD_DIM / 32; ++b) {
        const float d = ggml_fp16_to_fp32(blk[b].d);

        for (uint32_t j = 0; j < 32; ++j) {
            dst[b * 32 + j] = (float) blk[b].qs[j] * d;
        }
    }
}

void llama_engram_init(llama_context & lctx) {
    auto & en = lctx.engram;

    const llama_model & model = lctx.model;

    if (model.arch != LLM_ARCH_DEEPSEEK41) {
        return;
    }

    const auto & hparams = model.hparams;

    if (hparams.dsv4_engram_layer_count == 0) {
        return;
    }

    // the sidecar lives next to the GGUF: <dir>/<stem>.engram-constants.bin
    const std::string & mp = model.path;

    std::string dir  = ".";
    std::string stem = mp;

    const auto slash = mp.find_last_of('/');
    if (slash != std::string::npos) {
        dir  = mp.substr(0, slash);
        stem = mp.substr(slash + 1);
    }

    const auto dot = stem.find_last_of('.');
    if (dot != std::string::npos && dot > 0) {
        stem = stem.substr(0, dot);
    }

    const std::string sidecar = dir + "/" + stem + ".engram-constants.bin";

    if (!en.c.load(sidecar)) {
        LLAMA_LOG_WARN("engram: constants sidecar missing or invalid at '%s' - engram disabled "
                "(generate it with engram_constants.py)\n", sidecar.c_str());
        return;
    }

    // metadata-level validation only (shapes come from the GGUF header, so they are
    // valid even in a dry run); the buffer check happens at fill time
    bool ok = en.c.layers.size() == hparams.dsv4_engram_layer_count;

    for (uint32_t e = 0; ok && e < en.c.layers.size(); ++e) {
        const auto & cl = en.c.layers[e];

        if (cl.layer_id != hparams.dsv4_engram_layer_ids[e]) {
            LLAMA_LOG_WARN("engram: sidecar layer id %u != model layer id %u (entry %u)\n",
                    cl.layer_id, hparams.dsv4_engram_layer_ids[e], e);
            ok = false;
            break;
        }

        if (cl.layer_id >= model.layers.size()) {
            LLAMA_LOG_WARN("engram: sidecar layer id %u out of range\n", cl.layer_id);
            ok = false;
            break;
        }

        const auto & layer = model.layers[cl.layer_id];

        if (layer.engram_embd == nullptr || layer.engram_wkv == nullptr ||
            layer.engram_q    == nullptr || layer.engram_k    == nullptr) {
            LLAMA_LOG_WARN("engram: model layer %u is missing engram tensors\n", cl.layer_id);
            ok = false;
            break;
        }

        if (layer.engram_embd->ne[0] != llama_engram_constants::HEAD_DIM) {
            LLAMA_LOG_WARN("engram: table ne[0] = %d != %u\n",
                    (int) layer.engram_embd->ne[0], llama_engram_constants::HEAD_DIM);
            ok = false;
            break;
        }

        if ((uint64_t) layer.engram_embd->ne[1] != cl.rows) {
            // the prime partitioning must cover the table exactly
            LLAMA_LOG_WARN("engram: sidecar rows (%llu) != table rows (%d) for layer %u\n",
                    (unsigned long long) cl.rows, (int) layer.engram_embd->ne[1], cl.layer_id);
            ok = false;
            break;
        }
    }

    if (!ok) {
        LLAMA_LOG_WARN("engram: sidecar at '%s' does not match the model - engram disabled\n",
                sidecar.c_str());
        en.c = llama_engram_constants();
        return;
    }

    en.lookup.assign(en.c.layers.size(), nullptr);
    en.hist.clear();
    en.enabled = true;

    en.dump = getenv("LLAMA_ENGRAM_DUMP") != nullptr;
    LLAMA_LOG_INFO("engram: %zu layer(s) initialized from %s\n", en.c.layers.size(), sidecar.c_str());
}

ggml_tensor * llama_engram_new_lookup(llama_context & lctx, ggml_context * ctx, uint32_t il, int64_t n_tokens) {
    auto & en = lctx.engram;

    uint32_t e = 0;
    for (; e < en.c.layers.size(); ++e) {
        if (en.c.layers[e].layer_id == il) {
            break;
        }
    }
    GGML_ASSERT(e < en.c.layers.size()); // only call this on engram layers

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, llama_engram_constants::ROW_ELEMS, n_tokens);
    ggml_set_input(t);
    ggml_format_name(t, "engram_lookup_%u", e);

    if (en.lookup.size() <= e) {
        en.lookup.resize(e + 1, nullptr);
    }
    en.lookup[e] = t;

    return t;
}

bool llama_engram_prepare_inputs(llama_context & lctx, const llama_batch & ubatch) {
    auto & en = lctx.engram;

    if (!en.enabled || ubatch.token == nullptr) {
        return true;
    }

    const auto & c     = en.c;
    const auto & model = lctx.model;

    const uint32_t n_el  = (uint32_t) c.layers.size();
    const uint32_t n_tok = (uint32_t) ubatch.n_tokens;

    GGML_ASSERT(n_tok > 0);

    // the lookup inputs must exist and be sized for this batch: they are created
    // during the graph build (or belong to the reused graph)
    for (uint32_t e = 0; e < n_el; ++e) {
        if (e >= en.lookup.size() || en.lookup[e] == nullptr || en.lookup[e]->ne[1] != (int64_t) n_tok) {
            LLAMA_LOG_ERROR("engram: lookup input missing or sized for a different batch\n");
            return false;
        }
    }

    // ---- pass 1: commit every token to its sequence's tail and hash all rows.
    // An n-gram inside the batch looks back at tokens of this same batch, so the
    // tail is updated per token in position order. Nothing here touches the table,
    // so nothing can fault.
    en.rows.resize((size_t) n_el * n_tok * c.N_COLS);

    int32_t ids[c.N_GRAM];

    for (uint32_t i = 0; i < n_tok; ++i) {
        const llama_token   token = ubatch.token[i];
        const llama_seq_id  seq   = ubatch.seq_id[i][0];
        const llama_pos     pos   = ubatch.pos[i];

        GGML_ASSERT(token >= 0 && (uint32_t) token < c.n_vocab);

        const int32_t cid = c.token_map[token];

        auto it = en.hist.find(seq);
        if (it == en.hist.end()) {
            it = en.hist.emplace(seq, llama_engram_tail()).first;
        }
        auto & tail = it->second;

        // a token at position 0 starts a new prompt: no lookback
        if (pos == 0) {
            tail.v.fill(c.DEAD);
        }

        ids[0] = cid; // text-only: the current token is never masked/DEAD
        bool blocked = false;
        for (uint32_t g = 1; g < c.N_GRAM; ++g) {
            const int32_t id = tail.v[g - 1];

            blocked = blocked || id == c.DEAD;
            ids[g] = blocked ? c.pad_compressed : id;
        }

        for (uint32_t e = 0; e < n_el; ++e) {
            llama_engram_hash_row(c, e, ids, en.rows.data() + ((size_t) e * n_tok + i) * c.N_COLS);
        }

        // debug dump (LLAMA_ENGRAM_DUMP=1): token ids + computed rows for cross-checking
        if (en.dump && en.n_dumped < 64) {
            fprintf(stderr, "ENGDUMP pos=%d tok=%d cid=%d", (int) pos, (int) token, (int) cid);
            for (uint32_t e = 0; e < n_el; ++e) {
                const uint32_t * r = en.rows.data() + ((size_t) e * n_tok + i) * c.N_COLS;
                fprintf(stderr, " L%d=[", (int) c.layers[e].layer_id);
                for (uint32_t j = 0; j < c.N_COLS; ++j) {
                    fprintf(stderr, "%s%u", j ? "," : "", r[j]);
                }
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
            en.n_dumped++;
        }

        // shift: the tail becomes {cid, old[0], old[1]}
        for (uint32_t g = c.N_GRAM - 1; g > 1; --g) {
            tail.v[g - 1] = tail.v[g - 2];
        }
        tail.v[0] = cid;
    }

    // ---- prefetch: hand the OS every row range at once so the page faults go out
    // together instead of one per row on the thread the graph is waiting on
#if !defined(_WIN32) && defined(POSIX_MADV_WILLNEED)
    for (uint32_t e = 0; e < n_el; ++e) {
        const auto & layer = model.layers[c.layers[e].layer_id];
        const uint8_t * base = (const uint8_t *) layer.engram_embd->data;

        for (uint32_t i = 0; i < n_tok; ++i) {
            const uint32_t * r = en.rows.data() + ((size_t) e * n_tok + i) * c.N_COLS;

            for (uint32_t col = 0; col < c.N_COLS; ++col) {
                // madvise does not modify the range; the uintptr_t hop only silences -Wcast-qual
                posix_madvise((void *) (uintptr_t) (base + (size_t) r[col] * c.ROW_BYTES), c.ROW_BYTES, POSIX_MADV_WILLNEED);
            }
        }
    }
#endif

    // ---- pass 2: read + dequant into the F32 lookup inputs (one host write per layer)
    en.scratch.resize((size_t) c.ROW_ELEMS * n_tok);

    for (uint32_t e = 0; e < n_el; ++e) {
        const auto & layer = model.layers[c.layers[e].layer_id];

        // the gather reads ->data directly, so the tables must live in host memory
        // (force with -ot ".*engram.*=CPU"); buffers are only materialized outside
        // dry runs, hence this check lives here and not in llama_engram_init
        if (!ggml_backend_buffer_is_host(layer.engram_embd->buffer)) {
            LLAMA_LOG_ERROR("engram: engram_embd is not in a host buffer - pass -ot \".*engram.*=CPU\"\n");
            return false;
        }

        const uint8_t * base = (const uint8_t *) layer.engram_embd->data;

        float * dst = en.scratch.data();

        for (uint32_t i = 0; i < n_tok; ++i) {
            const uint32_t * r = en.rows.data() + ((size_t) e * n_tok + i) * c.N_COLS;
            float * tok = dst + (size_t) i * c.ROW_ELEMS;

            for (uint32_t col = 0; col < c.N_COLS; ++col) {
                llama_engram_dequant_row_q8_0(base + (size_t) r[col] * c.ROW_BYTES, tok + (size_t) col * c.HEAD_DIM);
            }
        }

        ggml_backend_tensor_set(en.lookup[e], dst, 0, (size_t) c.ROW_ELEMS * n_tok * sizeof(float));

        // debug dump: first token's dequantized rows, hex floats, for file-read cross-check
        if (en.dump && en.n_lookup_dumped < 2) {
            const uint32_t * r0 = en.rows.data() + ((size_t) e * n_tok + 0) * c.N_COLS;
            fprintf(stderr, "ENGDUMPLK e=%u tok0_rows", e);
            for (uint32_t col = 0; col < c.N_COLS; ++col) {
                fprintf(stderr, " %u", r0[col]);
            }
            fprintf(stderr, "\nENGDUMPLK e=%u tok0_vals", e);
            const float * v = dst;
            for (uint32_t k = 0; k < c.ROW_ELEMS; ++k) {
                union { float f; uint32_t u; } x; x.f = v[k];
                fprintf(stderr, " %08x", x.u);
            }
            fprintf(stderr, "\n");
            en.n_lookup_dumped++;
        }
    }

    return true;
}