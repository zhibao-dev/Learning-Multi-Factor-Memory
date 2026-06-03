#!/usr/bin/env bash
# Build a STANDALONE borge-audit wheel containing ONLY the thin audit slice
# (value model + embedder + signal extractor + audit/) — NOT the full agent or
# the paper1 self-FEP runtime. The slice is re-staged fresh from the research
# source on every build (no vendored copies in git → no divergence). The wheel
# is private (do NOT publish to public PyPI); ship it to paying pilots directly.
#
# Usage:  bash build_wheel.sh
# Output: packaging/borge-audit/dist/borge_audit-*.whl
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"
PY="${PY:-/Users/max_abel/opt/anaconda3/bin/python}"

rm -rf "$SRC" "$HERE/dist" "$HERE/build" "$HERE"/*.egg-info
mkdir -p "$SRC/borge/memory" "$SRC/borge/values" "$SRC/borge/affective" "$SRC/borge/audit"

# Empty package __init__ files. The REAL ones import unshipped modules
# (borge/__init__ -> agent, runner; memory/__init__ -> consolidation, forgetting;
# etc.), which would both break imports AND leak the research core. Empty inits
# keep the partial `borge` namespace importable with only the thin slice.
for d in borge borge/memory borge/values borge/affective; do
  : > "$SRC/$d/__init__.py"
done

# Thin slice — copied fresh from the research source (single source of truth).
cp "$ROOT/borge/memory/value.py"               "$SRC/borge/memory/value.py"
cp "$ROOT/borge/values/self_model.py"          "$SRC/borge/values/self_model.py"
cp "$ROOT/borge/affective/signal_extractor.py" "$SRC/borge/affective/signal_extractor.py"
cp "$ROOT"/borge/audit/*.py                    "$SRC/borge/audit/"

echo "== staged thin slice =="
find "$SRC" -name '*.py' | sed "s|$SRC/||" | sort

"$PY" -m pip wheel "$HERE" -w "$HERE/dist" --no-deps --no-build-isolation -q

WHL="$(ls -1 "$HERE/dist"/*.whl | sort | tail -1)"
echo "== wheel built: $(basename "$WHL") =="

# IP guard: the research core must be ABSENT from the wheel. Fail loudly if not.
"$PY" - "$WHL" <<'EOF'
import sys, zipfile
whl = sys.argv[1]
names = zipfile.ZipFile(whl).namelist()
forbidden = ("agent.py", "runner.py", "cli.py", "forgetting.py", "consolidation.py",
             "knowledge_graph.py", "cognitive_memory.py", "store.py", "retrieval.py",
             "value_net.py", "beliefs/", "inference/", "meta/")
leak = sorted(n for n in names if any(f in n for f in forbidden))
if leak:
    print("IP LEAK — these must not ship:", *leak, sep="\n  ")
    sys.exit(1)
shipped = sorted(n for n in names if n.endswith(".py"))
print("== IP guard OK — wheel ships only the thin slice ==")
for n in shipped:
    print("  ", n)
EOF
echo "Done. Private wheel: $WHL  (do NOT upload to public PyPI)"
