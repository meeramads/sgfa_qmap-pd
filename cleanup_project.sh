#!/bin/bash
# Cleanup script for sgfa_qmap-pd project
# Removes test files, logs, caches, and organizes config files

set -e  # Exit on error

echo "🧹 Starting project cleanup..."
echo ""

PROJECT_ROOT="/Users/meera/Desktop/sgfa_qmap-pd"
cd "$PROJECT_ROOT"

# Create backup directory for configs
echo "📦 Creating backup and archive directories..."
mkdir -p archived_configs
mkdir -p archived_test_outputs

# ============================================================================
# 1. Remove test output files from root
# ============================================================================
echo ""
echo "🗑️  Removing test output files from root..."

# Test plot images
rm -f test_distributions_converged.png
rm -f test_distributions_nonconverged.png
rm -f test_hyperparameter_posteriors_converged.png
rm -f test_hyperparameter_posteriors_nonconverged.png
rm -f test_hyperparameter_traces_converged.png
rm -f test_hyperparameter_traces_nonconverged.png
rm -f test_trace_converged.png
rm -f test_trace_nonconverged.png

# Test scripts
rm -f test_trace_plots.py
rm -f test_variance_profile.py
rm -f test_config_quick.py
rm -f test_synthetic_config.py

# Test debug script
rm -f debug_experiments.py

echo "   ✅ Removed test output files"

# ============================================================================
# 2. Move old configs to archive
# ============================================================================
echo ""
echo "📂 Archiving old/duplicate config files..."

# Move specific configs to archive
mv config_K2_no_sparsity.yaml archived_configs/ 2>/dev/null || true
mv config_K20_no_sparsity.yaml archived_configs/ 2>/dev/null || true
mv config_K50_no_sparsity.yaml archived_configs/ 2>/dev/null || true
mv config_debug.yaml archived_configs/ 2>/dev/null || true
mv config_K10_MAD5.0.yaml archived_configs/ 2>/dev/null || true
mv config_K10_full_voxels.yaml archived_configs/ 2>/dev/null || true
mv config_K15_full_voxels.yaml archived_configs/ 2>/dev/null || true
mv config_K20_MAD3.0.yaml archived_configs/ 2>/dev/null || true
mv config_K50_full_voxels.yaml archived_configs/ 2>/dev/null || true

echo "   ✅ Archived old config files to archived_configs/"

# ============================================================================
# 3. Clean up cache directories
# ============================================================================
echo ""
echo "🗄️  Removing cache directories..."

# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Mypy cache
rm -rf .mypy_cache

# Pytest cache
rm -rf .pytest_cache
find tests -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo "   ✅ Removed Python cache directories"

# ============================================================================
# 4. Clean up logs
# ============================================================================
echo ""
echo "📝 Cleaning up log files..."

# Remove old log files (keep logs directory structure)
if [ -d "logs" ]; then
    find logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
    echo "   ✅ Removed log files older than 7 days"
fi

# ============================================================================
# 5. Clean up test outputs directory
# ============================================================================
echo ""
echo "🧪 Archiving test_outputs..."

if [ -d "test_outputs" ]; then
    mv test_outputs/* archived_test_outputs/ 2>/dev/null || true
    rmdir test_outputs 2>/dev/null || true
    echo "   ✅ Archived test_outputs to archived_test_outputs/"
fi

# ============================================================================
# 6. Clean up .DS_Store files (macOS)
# ============================================================================
echo ""
echo "🍎 Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "   ✅ Removed .DS_Store files"

# ============================================================================
# 7. Summary
# ============================================================================
echo ""
echo "✨ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "   - Test files removed from root"
echo "   - Old configs archived to: archived_configs/"
echo "   - Test outputs archived to: archived_test_outputs/"
echo "   - Python caches cleared"
echo "   - Old logs cleaned"
echo ""
echo "Active configs remaining in root:"
ls -1 config_*.yaml 2>/dev/null | head -10 || echo "   (none)"
echo ""
echo "To restore archived configs: mv archived_configs/* ."
echo "To permanently delete archives: rm -rf archived_configs archived_test_outputs"
