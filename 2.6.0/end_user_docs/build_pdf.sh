#!/bin/bash
# Build FM2PROF PDF documentation with version number
# This script fetches the current version, updates the LaTeX file, and compiles the PDF

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FM2PROF PDF Build Script ===${NC}"

# Change to the source directory
cd "$(dirname "$0")/source"
echo "Working directory: $(pwd)"

# Step 1: Fetch the current version from pyproject.toml
echo -e "\n${YELLOW}Step 1: Fetching version from pyproject.toml${NC}"

# Navigate up to find pyproject.toml (from docs/end-user-docs/source to project root)
PYPROJECT_FILE="../../../pyproject.toml"

if [ ! -f "$PYPROJECT_FILE" ]; then
    echo -e "${RED}Error: pyproject.toml not found at $PYPROJECT_FILE${NC}"
    exit 1
fi

# Extract version using grep and sed
VERSION=$(grep -E '^version\s*=' "$PYPROJECT_FILE" | sed -E 's/version\s*=\s*"([^"]+)".*/\1/')

if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Could not extract version from pyproject.toml${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found version: $VERSION${NC}"

# Step 2: Update the LaTeX file with the version
echo -e "\n${YELLOW}Step 2: Updating version in LaTeX files${NC}"

TEX_FILES=("fm2prof_user_manual.tex" "fm2prof-release-notes.tex")

for TEX_FILE in "${TEX_FILES[@]}"; do
    if [ -f "$TEX_FILE" ]; then
        echo "Updating $TEX_FILE..."
        
        # Backup the original file
        cp "$TEX_FILE" "${TEX_FILE}.bak"
        
        # Replace the version line
        sed -i 's/\r//' "$TEX_FILE"
        sed -i 's/\\newcommand{\\fmprofversion}{[^}]*}/\\newcommand{\\fmprofversion}{'"$VERSION"'}/' "$TEX_FILE"

        # Remove the .tmp file created by sed on some systems
        rm -f "${TEX_FILE}.tmp"
        
        # Verify the replacement
        if grep -q "\\newcommand{\\\\fmprofversion}{$VERSION}" "$TEX_FILE"; then
            echo -e "${GREEN}✓ Version updated to $VERSION in $TEX_FILE${NC}"
        else
            echo -e "${RED}✗ Failed to update version in $TEX_FILE${NC}"
            
            # Restore backup
            mv "${TEX_FILE}.bak" "$TEX_FILE"
            exit 1
        fi
        
        # Remove backup
        rm -f "${TEX_FILE}.bak"
    else
        echo -e "${YELLOW}⚠ $TEX_FILE not found, skipping${NC}"
    fi
done

# Step 3: Compile PDFs with pdflatex and bibtex
echo -e "\n${YELLOW}Step 3: Compiling PDFs${NC}"

compile_pdf() {
    local tex_file=$1
    local base_name=$(basename "$tex_file" .tex)
    
    echo -e "\n${GREEN}Compiling $tex_file...${NC}"
    
    # First pdflatex run
    echo "  → First pdflatex pass..."
    pdflatex -interaction=nonstopmode "$tex_file" > "${base_name}_build.log" 2>&1 || {
        echo -e "${RED}✗ First pdflatex pass failed${NC}"
        echo "Last 20 lines of log:"
        tail -20 "${base_name}_build.log"
        return 1
    }
    
    # Check if bibtex is needed
    if grep -q '\\citation' "${base_name}.aux" 2>/dev/null; then
        echo "  → Running bibtex..."
        bibtex "$base_name" >> "${base_name}_build.log" 2>&1 || {
            echo -e "${YELLOW}⚠ bibtex warnings (continuing)${NC}"
        }
        
        # Second pdflatex run (after bibtex)
        echo "  → Second pdflatex pass (post-bibtex)..."
        pdflatex -interaction=nonstopmode "$tex_file" >> "${base_name}_build.log" 2>&1 || {
            echo -e "${RED}✗ Second pdflatex pass failed${NC}"
            tail -20 "${base_name}_build.log"
            return 1
        }
    else
        echo "  → No citations found, skipping bibtex"
    fi
    
    # Third pdflatex run (resolve references)
    echo "  → Third pdflatex pass (resolve references)..."
    pdflatex -interaction=nonstopmode "$tex_file" >> "${base_name}_build.log" 2>&1 || {
        echo -e "${RED}✗ Third pdflatex pass failed${NC}"
        tail -20 "${base_name}_build.log"
        return 1
    }
    
    # Final pdflatex run (ensure everything is correct)
    echo "  → Final pdflatex pass..."
    pdflatex -interaction=nonstopmode "$tex_file" >> "${base_name}_build.log" 2>&1 || {
        echo -e "${RED}✗ Final pdflatex pass failed${NC}"
        tail -20 "${base_name}_build.log"
        return 1
    }
    
    # Check if PDF was created
    if [ -f "${base_name}.pdf" ]; then
        echo -e "${GREEN}✓ Successfully created ${base_name}.pdf${NC}"
        return 0
    else
        echo -e "${RED}✗ PDF file was not created${NC}"
        return 1
    fi
}

# Compile both PDFs
SUCCESS=true

for TEX_FILE in "${TEX_FILES[@]}"; do
    if [ -f "$TEX_FILE" ]; then
        if ! compile_pdf "$TEX_FILE"; then
            SUCCESS=false
        fi
    fi
done

# Step 4: Report results
echo -e "\n${YELLOW}Step 4: Build Summary${NC}"

echo -e "\n${GREEN}Generated PDF files:${NC}"
ls -lh *.pdf 2>/dev/null || echo "No PDF files found"

if [ "$SUCCESS" = true ]; then
    echo -e "\n${GREEN}✓✓✓ All PDFs compiled successfully! ✓✓✓${NC}"
    echo -e "${GREEN}Version: $VERSION${NC}"
    exit 0
else
    echo -e "\n${RED}✗✗✗ Some PDFs failed to compile ✗✗✗${NC}"
    exit 1
fi
