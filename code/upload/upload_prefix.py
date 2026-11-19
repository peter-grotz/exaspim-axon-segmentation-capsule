"""Copy local folders into a new S3 prefix without ever overwriting or deleting anything.

Each SOURCE is ``<local dir>=<sub path under DEST>`` (``=`` alone for the prefix root). The run
stops before writing anything when DEST already holds any object, and stops again if a target
appears while copying. Only ``put`` is used: no sync, no delete. Writes
/results/upload/manifest.json (every file with its size) and verifies each upload's size afterwards.

Modes ``add-dry`` / ``add`` allow a prefix that already holds files, but stop before writing if any
planned target exists.

Usage: python upload_prefix.py <s3://bucket/prefix> <dry|write|add-dry|add> <SOURCE> [<SOURCE> ...]
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import s3fs

MODES = ("dry", "write", "add-dry", "add")
dest, mode, specs = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
if not dest.startswith("s3://") or dest.count("/") < 3 or mode not in MODES or not specs:
    sys.exit(__doc__)
fs = s3fs.S3FileSystem()
plan = []
for spec in specs:
    src, _, sub = spec.partition("=")
    root = Path(src)
    if not root.is_dir():
        sys.exit(f"source is not a directory: {root}")
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        plan.append((f, "/".join(x for x in (dest, sub.strip("/"), rel) if x), f.stat().st_size))
targets = [t for _, t, _ in plan]
if len(set(targets)) != len(targets):
    sys.exit("two sources map to the same target")
out = Path(os.environ.get("UPLOAD_OUT", "/results/upload"))
out.mkdir(parents=True, exist_ok=True)
manifest = {"dest": dest, "mode": mode, "files": len(plan), "bytes": sum(s for *_, s in plan),
            "items": [{"source": str(f), "target": t, "size": s} for f, t, s in plan]}  # fmt: skip
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"{len(plan)} files, {manifest['bytes'] / 1e9:.2f} GB -> {dest}/", flush=True)

existing = fs.find(dest + "/") + ([dest] if fs.isfile(dest) else [])
if mode.startswith("add"):  # the prefix may hold files, but none of the planned targets
    taken = sorted(set(t.removeprefix("s3://") for t in targets) & set(existing))
    if taken:
        print(f"STOP: {len(taken)} targets already exist, e.g. {taken[:5]}")
        print("nothing written")
        sys.exit(3)
    print(f"destination holds {len(existing)} objects; none of the {len(plan)} targets exist")
elif existing:  # never touch a prefix that already has content
    print(f"STOP: {dest}/ already holds {len(existing)} objects, e.g. {existing[:5]}")
    print("nothing written")
    sys.exit(3)
else:
    print("destination is empty")
if mode.endswith("dry"):
    print("dry run: nothing written")
    sys.exit(0)

clash = []


def put(item):
    """Upload one file unless its target exists (then record the clash and stop the run)."""
    f, t, _ = item
    if clash:
        return
    if fs.exists(t):  # appeared since the check: stop rather than overwrite
        clash.append(t)
        return
    fs.put_file(str(f), t)


with ThreadPoolExecutor(16) as pool:
    for i, _ in enumerate(pool.map(put, plan), 1):
        if i % 2000 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)} done", flush=True)
if clash:
    print(f"STOP: {clash[0]} appeared during the copy; it was not overwritten")
    sys.exit(4)
fs.invalidate_cache()
bad = [t for _, t, s in plan if fs.size(t) != s]
manifest["verified"] = not bad
manifest["size_mismatch"] = bad
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("verified: all sizes match" if not bad else f"SIZE MISMATCH on {len(bad)} files: {bad[:5]}")
sys.exit(0 if not bad else 5)
