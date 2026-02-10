# Makefile for epylabel paper reproduction
#
# Usage:
#   make setup      - Install all dependencies
#   make all        - Generate labels and plots
#   make labels     - Only generate labels
#   make plots      - Only generate plots (requires labels)
#   make test       - Run tests
#   make clean      - Remove generated output
#   make help       - Show this help message

.PHONY: all setup labels plots test clean help check

# Default target
all: labels plots

# Show help
help:
	@echo "Epylabel Paper Reproduction"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make setup      - Install all dependencies"
	@echo "  make all        - Generate labels and plots"
	@echo "  make labels     - Only generate labels"
	@echo "  make plots      - Only generate plots (requires labels)"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove generated output"
	@echo "  make check      - Check dependencies"
	@echo "  make help       - Show this help message"

# Check dependencies
check:
	python reproduce_paper.py --check

# Setup dependencies
setup:
	chmod +x setup_dependencies.sh
	./setup_dependencies.sh

# Generate labels
labels:
	python reproduce_paper.py --labels

# Generate plots (requires labels)
plots:
	python reproduce_paper.py --plots

# Run tests
test:
	pytest test/ -v

# Clean output
clean:
	rm -rf output/
	@echo "Cleaned output directory"

# Run the original scripts (for comparison)
original-labels:
	python paper_labels.py

original-plots:
	python paper_plots.py
