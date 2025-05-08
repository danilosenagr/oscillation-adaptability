#!/bin/bash

# compile_paper.sh - Script to compile LaTeX paper with multiple passes
# 
# This script compiles a LaTeX paper multiple times to ensure all references,
# citations, and cross-references are properly resolved. It runs in non-stop mode
# to prevent interruptions due to errors, and shows any remaining errors at the end.

# Set the paper name (without extension)
PAPER_NAME="oscillation_adaptability"
PAPER_DIR="paper"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Change to the paper directory
cd "$PAPER_DIR" || { print_error "Could not change to directory $PAPER_DIR"; exit 1; }

# Clean up auxiliary files
print_header "Cleaning up auxiliary files"
rm -f "$PAPER_NAME.aux" "$PAPER_NAME.bbl" "$PAPER_NAME.blg" "$PAPER_NAME.log" "$PAPER_NAME.out" "$PAPER_NAME.toc"
print_success "Auxiliary files removed"

# First LaTeX run
print_header "First LaTeX run"
pdflatex -interaction=nonstopmode "$PAPER_NAME.tex"
if [ $? -eq 0 ]; then
    print_success "First LaTeX run completed"
else
    print_warning "First LaTeX run completed with warnings/errors"
fi

# Run BibTeX
print_header "Running BibTeX"
bibtex "$PAPER_NAME"
if [ $? -eq 0 ]; then
    print_success "BibTeX run completed"
else
    print_warning "BibTeX run completed with warnings/errors"
fi

# Second LaTeX run
print_header "Second LaTeX run"
pdflatex -interaction=nonstopmode "$PAPER_NAME.tex"
if [ $? -eq 0 ]; then
    print_success "Second LaTeX run completed"
else
    print_warning "Second LaTeX run completed with warnings/errors"
fi

# Third LaTeX run
print_header "Third LaTeX run"
pdflatex -interaction=nonstopmode "$PAPER_NAME.tex"
if [ $? -eq 0 ]; then
    print_success "Third LaTeX run completed"
else
    print_warning "Third LaTeX run completed with warnings/errors"
fi

# Check for errors and warnings in the log file
print_header "Checking for errors and warnings"

# Check for errors
ERROR_COUNT=$(grep -c "^!" "$PAPER_NAME.log")
if [ "$ERROR_COUNT" -gt 0 ]; then
    print_error "Found $ERROR_COUNT errors in the log file:"
    grep -n "^!" "$PAPER_NAME.log" | head -10
    if [ "$ERROR_COUNT" -gt 10 ]; then
        echo "... and $(($ERROR_COUNT - 10)) more errors"
    fi
else
    print_success "No LaTeX errors found"
fi

# Check for undefined references
UNDEF_REF_COUNT=$(grep -c "Reference.*undefined" "$PAPER_NAME.log")
if [ "$UNDEF_REF_COUNT" -gt 0 ]; then
    print_warning "Found $UNDEF_REF_COUNT undefined references:"
    grep -n "Reference.*undefined" "$PAPER_NAME.log"
else
    print_success "No undefined references found"
fi

# Check for undefined citations
UNDEF_CITE_COUNT=$(grep -c "Citation.*undefined" "$PAPER_NAME.log")
if [ "$UNDEF_CITE_COUNT" -gt 0 ]; then
    print_warning "Found $UNDEF_CITE_COUNT undefined citations:"
    grep -n "Citation.*undefined" "$PAPER_NAME.log"
else
    print_success "No undefined citations found"
fi

# Check for hyperref warnings
HYPERREF_WARNING_COUNT=$(grep -c "Package hyperref Warning" "$PAPER_NAME.log")
if [ "$HYPERREF_WARNING_COUNT" -gt 0 ]; then
    print_warning "Found $HYPERREF_WARNING_COUNT hyperref warnings:"
    grep -n "Package hyperref Warning" "$PAPER_NAME.log" | head -10
    if [ "$HYPERREF_WARNING_COUNT" -gt 10 ]; then
        echo "... and $(($HYPERREF_WARNING_COUNT - 10)) more hyperref warnings"
    fi
else
    print_success "No hyperref warnings found"
fi

# Check for overfull boxes
OVERFULL_COUNT=$(grep -c "Overfull" "$PAPER_NAME.log")
if [ "$OVERFULL_COUNT" -gt 0 ]; then
    print_warning "Found $OVERFULL_COUNT overfull boxes:"
    grep -n "Overfull" "$PAPER_NAME.log" | head -10
    if [ "$OVERFULL_COUNT" -gt 10 ]; then
        echo "... and $(($OVERFULL_COUNT - 10)) more overfull boxes"
    fi
else
    print_success "No overfull boxes found"
fi

# Final summary
print_header "Compilation Summary"
if [ "$ERROR_COUNT" -eq 0 ] && [ "$UNDEF_REF_COUNT" -eq 0 ] && [ "$UNDEF_CITE_COUNT" -eq 0 ]; then
    print_success "Paper compiled successfully with no critical issues!"
    if [ "$HYPERREF_WARNING_COUNT" -gt 0 ] || [ "$OVERFULL_COUNT" -gt 0 ]; then
        print_warning "There are some minor warnings that could be addressed for a perfect document"
    fi
    echo -e "\nThe PDF file is available at: ${BLUE}$PAPER_NAME.pdf${NC}"
else
    print_error "Paper compiled with issues that need to be fixed"
    echo -e "\nPlease address the errors and warnings listed above"
fi

# Return to the original directory
cd - > /dev/null
