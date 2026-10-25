#!/bin/sh
# Render docs/report/report-draft.md to docs/report/report.pdf (A4, must stay at 2 pages).
# Needs pandoc and Google Chrome.
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
pandoc docs/report/report-draft.md -s --embed-resources -c docs/report/report.css -M pagetitle="Name That LLM" -o "$tmp/report.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=docs/report/report.pdf "$tmp/report.html" 2>/dev/null
python3 -c "import re; d = open('docs/report/report.pdf', 'rb').read(); print('pages:', len(re.findall(rb'/Type\s*/Page[^s]', d)))"
