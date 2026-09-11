#!/usr/bin/env python3
"""Scratch: dump GGUF metadata + tensor name patterns from an existing DSV4 GGUF."""
import re
import sys

import gguf


def dump(path: str, tensor_grep: str | None = None) -> None:
    r = gguf.GGUFReader(path)
    print(f"########## {path} ##########")
    print("==== structural metadata ====")
    wanted = (
        "deepseek4", "indexer", "compress", "hyper", "swiglu", "sliding",
        "hash", "nextn", "output_lora", "output_group", "head_count",
        "key_length", "value_length", "expert", "rope", "dflash",
    )
    for kv in r.fields.values():
        if not any(s in kv.name for s in wanted):
            continue
        v = kv.parts[kv.data[0]] if kv.data else None
        try:
            val = v.tolist() if hasattr(v, "tolist") else v
            if isinstance(val, list) and len(val) > 12:
                val = f"[{len(val)} entries] {val[:8]}..."
            print(f"  {kv.name} = {val}")
        except Exception as e:  # noqa: BLE001
            print(f"  {kv.name} = <err {e}>")
    print()
    print("==== tensor name patterns ====")
    pats: dict[str, tuple[str, tuple]] = {}
    for t in r.tensors:
        if tensor_grep and tensor_grep not in t.name.decode():
            continue
        p = re.sub(r"\.\d+\.", ".N.", t.name.decode())
        p = re.sub(r"experts\.\d+", "experts.X", p)
        if p not in pats:
            pats[p] = (str(t.tensor_type), tuple(t.shape))
    for p, (dt, sh) in sorted(pats.items()):
        print(f"  {dt:8s} {str(sh):24s} {p}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dump(p)