#pragma once

// DeepSeek-V4.1 engram: conditional memory read from hashed n-gram tables.
//
// A position is hashed as (max_ngram - 1) n-grams (2/3/4-gram), each split over
// n_head heads, so every (n-gram size, head) pair owns its own prime-sized bucket
// range inside the layer table. The hash constants are deterministic but their
// derivation needs the tokenizer and a primality test, so they are precomputed by
// engram_constants.py into a sidecar binary ("<model_stem>.engram-constants.bin"
// next to the GGUF) that this module loads; the GGUF carries no hash constants.
//
// The tables are ~97 GiB of Q8_0 per layer and only 24 rows are read per token, so
// the gather + dequantization run on the host in two passes (hash everything
// first, posix_madvise(WILLNEED) the row ranges, then read) and the graph receives
// the result as a plain F32 input tensor [24*256, n_tokens] per engram layer.

#include "llama.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_context;
struct ggml_tensor;
struct ggml_context;

struct llama_engram_constants {
    static constexpr uint32_t N_GRAM    = 4;    // engram.max_ngram_size
    static constexpr uint32_t N_HEAD    = 8;    // engram.head_count
    static constexpr uint32_t N_COLS    = (N_GRAM - 1) * N_HEAD; // 24 rows per token per layer
    static constexpr uint32_t HEAD_DIM  = 256;  // engram.key_length
    static constexpr uint32_t ROW_ELEMS = N_COLS * HEAD_DIM;     // 6144 (wkv input width)
    static constexpr uint32_t ROW_BYTES = (HEAD_DIM / 32) * (2 + 32); // Q8_0: 8 x (fp16 scale + 32 i8) = 272
    static constexpr int32_t  DEAD      = -1;   // blocked lookback sentinel

    struct layer {
        uint32_t layer_id        = 0;
        int64_t  mults[N_GRAM]   = {};  // odd; mults[g] multiplies the id g positions back
        uint32_t primes[N_COLS]  = {};  // bucket modulus per (n-gram, head) column
        uint32_t offsets[N_COLS] = {};  // first row of each bucket range (cumsum)
        uint64_t rows            = 0;   // total table rows == sum of the primes
    };

    std::vector<layer>   layers;
    int32_t              pad_compressed   = 0;  // compressed id filling blocked lookback slots
    uint32_t             compressed_vocab = 0;
    uint32_t             n_vocab          = 0;
    std::vector<int32_t> token_map;              // [n_vocab] token id -> compressed id

    // Parse + validate the sidecar (format documented in engram_constants.py).
    // Returns false on any structural or consistency failure.
    bool load(const std::string & path);
};

// ids[0] = current token's compressed id, ids[g] = the id g positions back (already
// padded/blocked). Writes N_COLS row indices for engram layer e.
void llama_engram_hash_row(const llama_engram_constants & c, uint32_t e, const int32_t * ids, uint32_t * rows);

// Dequantize one Q8_0 table row (HEAD_DIM values) into dst.
void llama_engram_dequant_row_q8_0(const uint8_t * row, float * dst);

// Per-sequence tail of the last N_GRAM-1 committed compressed ids.
struct llama_engram_tail {
    std::array<int32_t, llama_engram_constants::N_GRAM - 1> v;

    llama_engram_tail() { v.fill(llama_engram_constants::DEAD); }
};

// Runtime state owned by llama_context.
struct llama_engram_runtime {
    bool enabled = false;

    // debug: dump per-token hash rows to stderr (LLAMA_ENGRAM_DUMP=1)
    bool     dump     = false;
    uint32_t n_dumped = 0;
    uint32_t n_lookup_dumped = 0;

    llama_engram_constants c;

    // F32 [ROW_ELEMS, n_tokens] graph input per engram layer, created during each
    // graph build (llama_engram_new_lookup) and filled by llama_engram_prepare_inputs.
    std::vector<ggml_tensor *> lookup;

    // per-sequence compressed-id history (see llama_engram_tail)
    std::unordered_map<llama_seq_id, llama_engram_tail> hist;

    // fill scratch, sized on demand and kept across batches
    std::vector<uint32_t> rows;     // [n_layers][n_tokens][N_COLS]
    std::vector<float>    scratch;  // [ROW_ELEMS * n_tokens]
};

// Context-construction init: for DEEPSEEK41 with engram layers, load the constants
// sidecar next to the model file; warn + leave the module disabled when absent.
void llama_engram_init(llama_context & lctx);

// Graph-build helper: create the F32 lookup input for engram layer `il` (block id)
// and stash it on the context. Call once per engram layer per graph build.
ggml_tensor * llama_engram_new_lookup(llama_context & lctx, ggml_context * ctx, uint32_t il, int64_t n_tokens);

// Per-batch fill: commit tokens to the per-seq history, hash all rows, prefetch,
// then gather + dequant into the lookup inputs. Call after the graph build (or
// reuse) with the batch about to run, before the graph is launched.
bool llama_engram_prepare_inputs(llama_context & lctx, const llama_batch & ubatch);