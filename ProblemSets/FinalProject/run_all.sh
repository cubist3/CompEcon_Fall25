#!/bin/bash
#==============================================================================
# ECON 833 Final Project: Tax Capitalization in UK Used Car Market
# Run this script to reproduce all results
#==============================================================================

echo "ECON 833 Final Project"
echo "Tax Capitalization in UK Used Car Market"

# data
if [ ! -f "toyota.csv" ]; then
    echo "ERROR: toyota.csv not found!"
    echo "Please ensure the data file is in the directory."
    exit 1
fi

# Create output directories
mkdir -p output/figures
mkdir -p output/tables

# Run main analysis
echo ""
echo "[1/2] Running Python (main_analysis.py)"
echo "note: bootstrap might takes several minutes"
python3 main_analysis.py

# Check if analysis succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Python analysis failed!"
    exit 1
fi

# Compile LaTeX
echo ""
echo "compiling LaTeX document"
pdflatex -interaction=nonstopmode FinalProject_Cho.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode FinalProject_Cho.tex > /dev/null 2>&1

# Check if PDF was created
if [ -f "FinalProject_Cho.pdf" ]; then
    echo "      PDF compiled successfully!"
else
    echo "      Warning: PDF compilation may have issues. Check .log file."
fi

# Clean up auxiliary files
echo ""
echo "Cleaning up..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo ""
echo "complete"
echo ""
echo "Output files:"
echo "  - FinalProject_Cho.pdf"
echo "  - output/figures/ (All figures)"
echo "  - output/tables/ (All tables)"
echo ""
