#!/usr/bin/env bash
# run_ps8.sh

set -euo pipefail

###############################################################################
# relative path
###############################################################################
cd "$(dirname "$0")"
echo "Running Problem Set 8 pipeline from: $(pwd)"

###############################################################################
# run the Python SMM code (main.py)
# recreate smm_results.npz and any tables you write out.
###############################################################################
# Pick a Python executable that exists on the system
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "running SMM estimation via main.py"
$PYTHON main.py
echo "running python complete"

###############################################################################
# compile the tex ProblemSet8_Cho.tex to pdf
###############################################################################
TEXFILE="ProblemSet8_Cho.tex"

echo "compiling latex: ${TEXFILE}"

if command -v latexmk >/dev/null 2>&1; then
    echo "Using latexmk ..."
    latexmk -pdf -interaction=nonstopmode "${TEXFILE}"
    # optional cleanup of aux files:
    latexmk -c "${TEXFILE}"
else
    echo "latexmk not found. Falling back to pdflatex + bibtex ..."
    pdflatex -interaction=nonstopmode "${TEXFILE}" || true
    bibtex "${TEXFILE%.tex}" || true
    pdflatex -interaction=nonstopmode "${TEXFILE}" || true
    pdflatex -interaction=nonstopmode "${TEXFILE}" || true
fi

echo "complete"
echo "  $(pwd)/ProblemSet8_Cho.pdf"
