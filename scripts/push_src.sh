#!/usr/bin/env bash
# Upload src/nthl to the private Kaggle dataset nabilshah/nthl-src, so
# notebooks can import our code. The first run creates the dataset, later
# runs add a version named after the current commit.
#
#   scripts/push_src.sh [version message]
set -euo pipefail
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
OUT=build/nthl-src

rm -rf "$OUT" && mkdir -p "$OUT"
cp -R src/nthl "$OUT/nthl"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "nthl-src", "id": "nabilshah/nthl-src", "licenses": [{"name": "other"}]}
JSON

MSG="${1:-$(git rev-parse --short HEAD)}"
if $K datasets list --mine --search nthl-src 2>/dev/null | grep -q "^nabilshah/nthl-src "; then
  $K datasets version -p "$OUT" -r tar -m "$MSG"
else
  $K datasets create -p "$OUT" -r tar
fi
