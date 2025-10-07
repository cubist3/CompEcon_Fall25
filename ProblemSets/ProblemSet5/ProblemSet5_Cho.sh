#!/usr/bin/env bash
set -euo pipeful

Rscript ProblemSet5_Cho.R

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode ProblemSet5_Cho.tex
else
  pdflatex -interaction=nonstopmode ProblemSet5_Cho.tex || true
  pdflatex -interaction=nonstopmode ProblemSet5_Cho.tex || true
fi

echo "Done. PDF should be ProblemSet5_Cho.pdf"
