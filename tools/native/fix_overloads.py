#!/usr/bin/env python3
"""Find and patch the entries in node-mlx's InitOps Set() call that MLX 0.32 rejects.

Run from the root of a node-mlx checkout (MLX_PREFIX selects the MLX headers).

Taking the address of an overloaded MLX function is ambiguous once MLX grows overloads, and a
template-deduction failure shows up in clang only as "(no value)" in the deduced pack. This maps
that slot back to the source expression and applies an explicit static_cast.
"""
import os
import re
import subprocess
import sys

SRC = "src/ops.cc"
MLX_PREFIX = os.environ.get("MLX_PREFIX", "/opt/homebrew/opt/mlx")
FLAGS = (
    "-std=gnu++17 -fsyntax-only -I. -Inode_modules/node-api-headers/include -Ideps/kizunapi "
    f"-I{MLX_PREFIX}/include -D_DARWIN_USE_64_BIT_INODE=1 -D_LARGEFILE_SOURCE "
    "-D_FILE_OFFSET_BITS=64 -DBUILDING_NODE_EXTENSION -DNAPI_VERSION=9 -Wno-c++20-extensions"
).split()


def compile_ops():
    p = subprocess.run(["clang++", *FLAGS, SRC], capture_output=True, text=True)
    return p.stdout + p.stderr


def set_pairs(text):
    start = [i for i, l in enumerate(text.split("\n")) if "ki::Set(env, exports," in l and i > 700][0]
    lines, chunk, i = text.split("\n"), [], start
    while True:
        chunk.append(lines[i])
        if lines[i].rstrip().endswith(");"):
            break
        i += 1
    body = "\n".join(chunk)
    inner = body[body.index("ki::Set(env, exports,") + len("ki::Set(env, exports,") :]
    inner = inner.rsplit(");", 1)[0]
    args, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    args.append(cur.strip())
    args = [a for a in args if a]
    return [(args[k], args[k + 1]) for k in range(0, len(args) - 1, 2)]


def failing_slot(log):
    m = re.search(r"deduced incomplete pack <(.*?)> for template parameter", log, re.S)
    if not m:
        return None, None
    pack = m.group(1)
    items, depth, cur = [], 0, ""
    for ch in pack:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    items.append(cur.strip())
    for k in range(0, len(items)):
        if "no value" in items[k]:
            return k, items
    return None, items


text = open(SRC).read()
for attempt in range(20):
    log = compile_ops()
    if "error:" not in log:
        print(f"[{attempt}] ops.cc is clean")
        break
    slot, items = failing_slot(log)
    if slot is None:
        print("Could not map the failure; remaining errors:")
        print("\n".join(l for l in log.split("\n") if "error:" in l)[:800])
        sys.exit(1)
    pairs = set_pairs(text)
    # pack[0] is pairs[1].key, so value slot k (odd) is pairs[(k + 1) // 2]
    idx = (slot + 1) // 2 if slot % 2 else (slot + 2) // 2
    key, value = pairs[idx]
    print(f"[{attempt}] pack slot {slot} -> pair {idx}: {key} -> {value[:70]}")
    target = re.fullmatch(r"&mx::([A-Za-z_][A-Za-z0-9_]*)", value.strip())
    if not target:
        print("   (not a plain function address; stopping)")
        sys.exit(1)
    name = target.group(1)
    patched = (
        f'static_cast<mx::array (*)(const mx::array&, mx::StreamOrDevice)>(\n'
        f"              &mx::{name})"
    )
    text = text.replace(f"{key}, {value},", f"{key}, {patched},", 1)
    open(SRC, "w").write(text)