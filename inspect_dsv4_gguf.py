#!/usr/bin/env python3
"""Dump tensor names/dtypes and arch metadata from a DSv4-family GGUF (naming ground truth)."""
import re
import sys

import gguf

path = sys.argv[1]
r = gguf.GGUFReader(path)

print("==== metadata (structural keys) ====")
interesting = (
    "deepseek4", "indexer", "compress", "hyper_connection", "swiglu", "sliding",
    "hash", "nextn", "output_lora", "output_group", "head_count", "key_length",
    "value_length", "expert", "rope", "attention.causal", "architectures", "quant",
)
for field in r.fields.values():
    if not any(s in field.name for s in interesting):
        continue
    if field.data:
        v = field.parts[field.data[0]]
        try:
            val = v.tolist() if hasattr(v, "tolist") else v
        except Exception as e:  # noqa: BLE001
            val = f"<{e}>"
        if isinstance(val, list) and len(val) > 12:
            val = f"[{len(val)} entries] first={val[:6]}"
        print(f"  {field.name} = {val}")

print()
print("==== tensor name patterns (deduped) ====")
pats: dict[str, tuple[str, tuple]] = {}
for t in r.tensors:
    name = t.name.decode() if isinstance(t.name, bytes) else t.name
    p = re.sub(r"\.\d+\.", ".N.", name)
    p = re.sub(r"experts\.\d+", "experts.X", p)
    if p not in pats:
        pats[p] = (str(t.tensor_type), tuple(t.shape))
for p, (dt, sh) in sorted(pats.items()):
    print(f"  {dt:8s} {str(sh):24s} {p}")