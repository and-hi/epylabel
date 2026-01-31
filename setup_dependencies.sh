#!/bin/bash
#
# Setup script for epylabel dependencies
# This script installs all required Python and R packages
#
# Usage:
#   chmod +x setup_dependencies.sh
#   ./setup_dependencies.sh
#

set -e

echo "========================================"
echo "EPYLABEL - Dependency Setup Script"
echo "========================================"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
    echo "Note: Some commands may require sudo access."
else
    SUDO=""
fi

echo ""
echo "Step 1: Checking system requirements..."
echo "----------------------------------------"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "Python version: $PYTHON_VERSION"
else
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Check if R is installed
if command -v R &> /dev/null; then
    R_VERSION=$(R --version | head -n1)
    echo "R version: $R_VERSION"
else
    echo "R is not installed. Attempting to install..."
    if command -v apt-get &> /dev/null; then
        $SUDO apt-get update
        $SUDO apt-get install -y r-base r-base-dev libtirpc-dev
    elif command -v yum &> /dev/null; then
        $SUDO yum install -y R R-devel
    elif command -v brew &> /dev/null; then
        brew install r
    else
        echo "ERROR: Could not install R. Please install R manually."
        exit 1
    fi
fi

echo ""
echo "Step 2: Installing Python dependencies..."
echo "----------------------------------------"

# Install Python packages
pip install --upgrade pip

# Core packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Data handling
pip install pyarrow fastparquet

# Geospatial
pip install geopandas shapely

# Visualization
pip install plotly dash

# R integration
pip install rpy2

# Progress bars
pip install tqdm

# Testing
pip install pytest

echo ""
echo "Step 3: Installing R packages..."
echo "----------------------------------------"

# Install R packages
R -e '
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install Rcpp first (required by bcp)
if (!require("Rcpp", quietly = TRUE)) {
    install.packages("Rcpp")
}

# Install RcppArmadillo (required by bcp)
if (!require("RcppArmadillo", quietly = TRUE)) {
    install.packages("RcppArmadillo")
}

# Try to install bcp
tryCatch({
    if (!require("bcp", quietly = TRUE)) {
        # Try from CRAN first
        install.packages("bcp")
    }
}, error = function(e) {
    # If CRAN fails, try archive version
    message("Installing bcp from archive...")
    install.packages("https://cran.r-project.org/src/contrib/Archive/bcp/bcp_4.0.3.tar.gz",
                     repos = NULL, type = "source")
})

# Verify installation
if (require("bcp", quietly = TRUE)) {
    message("SUCCESS: bcp package is installed")
} else {
    stop("ERROR: Failed to install bcp package")
}
'

echo ""
echo "Step 4: Verifying installation..."
echo "----------------------------------------"

python3 -c "
import sys
errors = []

packages = ['pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib',
            'seaborn', 'geopandas', 'pyarrow', 'tqdm', 'rpy2']

for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ✗ {pkg}')
        errors.append(pkg)

# Test rpy2 and bcp
try:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    bcp = rpackages.importr('bcp')
    print('  ✓ R bcp package')
except Exception as e:
    print(f'  ✗ R bcp package: {e}')
    errors.append('R bcp')

if errors:
    print(f'\nWARNING: Some packages could not be installed: {errors}')
    sys.exit(1)
else:
    print('\n✓ All dependencies installed successfully!')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "You can now run:"
echo "  python reproduce_paper.py"
echo "========================================"
