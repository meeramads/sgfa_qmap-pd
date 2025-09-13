#!/bin/bash
# Remote Workstation Setup Script with Quota and CUDA Management
# Handles environment setup, CUDA configuration, and quota checks
# Run experiments separately with: python remote_workstation/run_experiments.py

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_section() {
    echo -e "\n${MAGENTA}═══════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}   $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}\n"
}

# Function to check disk quota
check_quota() {
    print_section "Disk Quota Check"
    
    # Check quota if available
    if command -v quota &> /dev/null; then
        print_info "Current disk quota:"
        quota -s || true
    else
        print_warning "quota command not available"
    fi
    
    # Check home directory usage
    print_info "Home directory usage:"
    df -h ~/ | tail -1
    
    # Check current directory usage
    print_info "Current directory usage:"
    du -sh . 2>/dev/null || echo "Unable to determine"
    
    # Check venv size if it exists
    if [ -d "venv" ]; then
        VENV_SIZE=$(du -sh venv/ | cut -f1)
        print_info "Virtual environment size: $VENV_SIZE"
    fi
    
    # Top space consumers in home directory
    print_info "Top 5 space consumers in home directory:"
    du -sh ~/* 2>/dev/null | sort -rh | head -5 || echo "Unable to determine"
    
    echo ""
    read -p "Continue with setup? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Setup cancelled. Consider cleaning up space first."
        exit 1
    fi
}

# Function to detect and configure CUDA
configure_cuda() {
    print_section "CUDA Configuration"
    
    # Try to detect CUDA version from nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        print_info "Detected CUDA Version from nvidia-smi: $CUDA_VERSION"
    else
        print_warning "nvidia-smi not found. Manual CUDA configuration may be needed."
        CUDA_VERSION="12.6"  # Default fallback
    fi
    
    # Find CUDA installation paths
    print_info "Searching for CUDA installations..."
    CUDA_PATHS=$(ls -d /opt/cuda/cuda-* 2>/dev/null || true)
    
    if [ -z "$CUDA_PATHS" ]; then
        # Try alternative locations
        CUDA_PATHS=$(ls -d /usr/local/cuda* 2>/dev/null || true)
    fi
    
    if [ -z "$CUDA_PATHS" ]; then
        print_error "No CUDA installations found in standard locations"
        print_info "Please specify CUDA path manually"
        read -p "Enter CUDA installation path (or press Enter to skip): " CUSTOM_CUDA_PATH
        if [ ! -z "$CUSTOM_CUDA_PATH" ] && [ -d "$CUSTOM_CUDA_PATH" ]; then
            CUDA_HOME="$CUSTOM_CUDA_PATH"
        else
            print_warning "Proceeding without CUDA configuration"
            return 1
        fi
    else
        print_info "Found CUDA installations:"
        echo "$CUDA_PATHS"
        
        # Try to match detected version or use latest
        CUDA_HOME=$(echo "$CUDA_PATHS" | grep "$CUDA_VERSION" | head -1)
        if [ -z "$CUDA_HOME" ]; then
            CUDA_HOME=$(echo "$CUDA_PATHS" | tail -1)
        fi
    fi
    
    print_status "Using CUDA installation: $CUDA_HOME"
    
    # Export CUDA environment variables
    export CUDA_HOME="$CUDA_HOME"
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
    
    print_info "CUDA environment variables set"
    
    # Add to bashrc if not already present
    if ! grep -q "CUDA_HOME=${CUDA_HOME}" ~/.bashrc 2>/dev/null; then
        print_info "Adding CUDA configuration to ~/.bashrc"
        {
            echo ""
            echo "# CUDA configuration for sgfa_qmap-pd"
            echo "export CUDA_HOME=${CUDA_HOME}"
            echo "export LD_LIBRARY_PATH=/usr/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH"
            echo "export PATH=${CUDA_HOME}/bin:\$PATH"
            echo "export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_HOME}"
        } >> ~/.bashrc
        print_status "CUDA configuration added to ~/.bashrc"
    else
        print_info "CUDA configuration already in ~/.bashrc"
    fi
    
    # Verify CUDA libraries
    print_info "Verifying CUDA libraries..."
    if [ -f "${CUDA_HOME}/lib64/libcudart.so" ] || [ -f "${CUDA_HOME}/lib/libcudart.so" ]; then
        print_status "CUDA runtime library found"
    else
        print_warning "CUDA runtime library not found at expected location"
    fi
    
    # Check for cuDNN
    CUDNN_FILES=$(find "${CUDA_HOME}" -name "libcudnn*" 2>/dev/null | head -1)
    if [ ! -z "$CUDNN_FILES" ]; then
        print_status "cuDNN library found"
    else
        print_warning "cuDNN library not found (may affect performance)"
    fi
}

# Function to clean up disk space
cleanup_space() {
    print_section "Disk Space Cleanup"
    
    print_info "Options for freeing disk space:"
    echo "  1) Clear pip cache"
    echo "  2) Remove NVIDIA pip packages (use system CUDA instead)"
    echo "  3) Clean old experiment results"
    echo "  4) Clean temporary files"
    echo "  5) Show large directories"
    echo "  6) Skip cleanup"
    echo ""
    read -p "Choose option (1-6): " cleanup_choice
    
    case $cleanup_choice in
        1)
            print_info "Clearing pip cache..."
            rm -rf ~/.cache/pip
            pip cache purge 2>/dev/null || true
            print_status "Pip cache cleared"
            ;;
        2)
            print_info "Removing NVIDIA pip packages..."
            pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 \
                nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
                nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
                nvidia-nvjitlink-cu12 -y 2>/dev/null || true
            print_status "NVIDIA packages removed"
            ;;
        3)
            if [ -d "remote_workstation_experiment_results" ]; then
                OLD_RESULTS=$(find remote_workstation_experiment_results -maxdepth 1 -type d -mtime +7 | wc -l)
                if [ "$OLD_RESULTS" -gt 0 ]; then
                    print_info "Found $OLD_RESULTS experiment results older than 7 days"
                    read -p "Remove old results? (y/n): " -n 1 -r
                    echo ""
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        find remote_workstation_experiment_results -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
                        print_status "Old results removed"
                    fi
                else
                    print_info "No old results to clean"
                fi
            fi
            ;;
        4)
            print_info "Cleaning temporary files..."
            rm -rf /tmp/jax_cache_* 2>/dev/null || true
            rm -rf ~/.cache/jax* 2>/dev/null || true
            print_status "Temporary files cleaned"
            ;;
        5)
            print_info "Large directories in home:"
            du -sh ~/* 2>/dev/null | sort -rh | head -10
            ;;
        6)
            print_info "Skipping cleanup"
            ;;
        *)
            print_warning "Invalid option, skipping cleanup"
            ;;
    esac
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "requirements.txt" ]; then
        print_error "Error: Run this script from the sgfa_qmap-pd directory"
        exit 1
    fi
}

# Function to setup Git for HTTPS
setup_git() {
    print_section "Git Configuration"
    
    # Check current remote URL
    CURRENT_URL=$(git remote get-url origin 2>/dev/null || echo "")
    
    if [[ $CURRENT_URL == *"git@github.com"* ]]; then
        print_info "Converting Git remote from SSH to HTTPS..."
        git remote set-url origin https://github.com/meeramads/sgfa_qmap-pd.git
        print_status "Git remote updated to HTTPS"
    elif [[ $CURRENT_URL == *"https://github.com"* ]]; then
        print_status "Git already configured for HTTPS"
    else
        print_warning "Unexpected Git configuration: $CURRENT_URL"
    fi
    
    print_info "To push changes, you'll need a GitHub Personal Access Token"
    print_info "Create one at: https://github.com/settings/tokens/new"
}

# Function to setup Python environment with quota awareness
setup_environment() {
    print_section "Python Environment Setup"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
        VENV_SIZE=$(du -sh venv/ | cut -f1)
        print_info "Current venv size: $VENV_SIZE"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Check if dependencies are installed
    if ! python -c "import jax" 2>/dev/null; then
        print_info "Installing dependencies (this may take a while)..."
        
        # Upgrade pip first
        pip install --upgrade pip
        
        # Install base requirements
        print_info "Installing base requirements..."
        pip install -r requirements.txt
        
        # Install JAX with local CUDA (doesn't download CUDA packages)
        print_info "Installing JAX with local CUDA support..."
        pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        
        print_status "Dependencies installed"
    else
        print_status "Dependencies already installed"
    fi
}

# Function to verify JAX with CUDA
verify_jax_cuda() {
    print_section "Verifying JAX and CUDA"
    
    print_info "Testing JAX with CUDA..."
    
    python -c "
import sys
import os

# Ensure CUDA paths are set
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/opt/cuda/cuda-12.6')
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '--xla_gpu_cuda_data_dir=' + os.environ['CUDA_HOME'])

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    
    print(f'JAX version: {jax.__version__}')
    print(f'NumPyro version: {numpyro.__version__}')
    print(f'JAX backend: {jax.lib.xla_bridge.get_backend().platform}')
    print(f'JAX devices: {jax.devices()}')
    
    # Test GPU computation
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices()):
        x = jnp.ones((100, 100))
        y = jnp.dot(x, x)
        y.block_until_ready()
        print('✓ GPU computation test successful')
    else:
        print('⚠ No GPU devices found')
        
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
" || {
    print_error "JAX/CUDA verification failed"
    print_info "Troubleshooting steps:"
    echo "  1. Check CUDA_HOME: echo \$CUDA_HOME"
    echo "  2. Check LD_LIBRARY_PATH: echo \$LD_LIBRARY_PATH"
    echo "  3. Verify CUDA installation: ls -la \$CUDA_HOME/lib64/"
    echo "  4. Try reinstalling JAX: pip uninstall jax jaxlib -y && pip install 'jax[cuda12_local]'"
    return 1
}
    
    print_status "JAX with CUDA verified successfully"
}

# Function to create necessary directories
create_directories() {
    print_info "Creating project directories..."
    mkdir -p remote_workstation_experiment_results
    mkdir -p checkpoints
    mkdir -p logs
    print_status "Directories created"
}

# Function to check GPU availability
check_gpu() {
    print_section "GPU Status"
    
    if command -v nvidia-smi &> /dev/null; then
        # Show GPU info
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
        
        # Get CUDA version from nvidia-smi
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_info "CUDA Version: $CUDA_VERSION"
        
        print_status "GPU detected and accessible"
    else
        print_warning "nvidia-smi not available - GPU may not be accessible"
    fi
}

# Function to check data directory
check_data() {
    print_section "Data Directory Check"
    
    # Check if data directory exists in config
    CONFIG_FILE="remote_workstation/config.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        DATA_DIR=$(grep "data_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
        
        if [ -d "$DATA_DIR" ]; then
            print_status "Data directory found: $DATA_DIR"
            FILE_COUNT=$(find "$DATA_DIR" -type f 2>/dev/null | wc -l)
            DIR_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
            print_info "  Contains $FILE_COUNT files"
            print_info "  Total size: $DIR_SIZE"
        else
            print_warning "Data directory not found: $DATA_DIR"
            print_warning "  Update the path in $CONFIG_FILE before running experiments"
            echo ""
            read -p "Enter the correct data directory path (or press Enter to skip): " NEW_DATA_DIR
            if [ ! -z "$NEW_DATA_DIR" ]; then
                if [ -d "$NEW_DATA_DIR" ]; then
                    sed -i.bak "s|data_dir:.*|data_dir: \"$NEW_DATA_DIR\"|" "$CONFIG_FILE"
                    print_status "Updated data directory in config"
                else
                    print_error "Directory $NEW_DATA_DIR does not exist"
                fi
            fi
        fi
    else
        print_warning "Config file not found: $CONFIG_FILE"
    fi
}

# Function to show how to run experiments
show_run_instructions() {
    print_section "Ready to Run Experiments!"
    
    echo "Your environment is set up. To run experiments:"
    echo ""
    echo "${GREEN}Option 1: Using tmux (recommended for long runs)${NC}"
    echo "  tmux new -s experiments"
    echo "  source venv/bin/activate"
    echo "  python remote_workstation/run_experiments.py --experiments all"
    echo "  # Detach with Ctrl+B, then D"
    echo ""
    echo "${GREEN}Option 2: Using nohup${NC}"
    echo "  source venv/bin/activate"
    echo "  nohup python remote_workstation/run_experiments.py --experiments all > experiment.log 2>&1 &"
    echo ""
    echo "${GREEN}Option 3: Direct run (stays attached)${NC}"
    echo "  source venv/bin/activate"
    echo "  python remote_workstation/run_experiments.py --experiments all"
    echo ""
    echo "${GREEN}To run specific experiments:${NC}"
    echo "  python remote_workstation/run_experiments.py --experiments data_validation"
    echo "  python remote_workstation/run_experiments.py --experiments method_comparison"
    echo ""
    echo "${GREEN}Monitor progress:${NC}"
    echo "  tail -f logs/remote_workstation_experiments.log"
    echo "  watch -n 1 nvidia-smi"
}

# Function to show status
show_status() {
    print_section "System Status"
    
    # Check disk usage
    print_info "Disk usage:"
    df -h ~/ | tail -1
    
    if [ -d "venv" ]; then
        VENV_SIZE=$(du -sh venv/ | cut -f1)
        print_info "Virtual environment size: $VENV_SIZE"
    fi
    
    # Check if experiments are running
    if pgrep -f "run_experiments.py" > /dev/null; then
        print_status "Experiments are RUNNING"
        ps aux | grep "[r]un_experiments.py" | head -1
    else
        print_info "No experiments currently running"
    fi
    
    # Show recent results
    if [ -d "remote_workstation_experiment_results" ] && [ "$(ls -A remote_workstation_experiment_results)" ]; then
        echo ""
        print_info "Recent experiment results:"
        ls -lht remote_workstation_experiment_results/ | head -5
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        print_info "GPU utilization:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    fi
    
    # Check CUDA configuration
    echo ""
    print_info "CUDA configuration:"
    echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
    echo "  XLA_FLAGS: ${XLA_FLAGS:-not set}"
}

# Function to run full setup
run_full_setup() {
    check_quota
    configure_cuda
    setup_environment
    verify_jax_cuda
    create_directories
    check_gpu
    check_data
    setup_git
    show_run_instructions
}

# Main menu
main_menu() {
    print_section "SGFA qMAP-PD Remote Workstation Setup"
    
    echo "Setup and configuration options:"
    echo ""
    echo "  ${GREEN}1)${NC} Full setup (recommended for first time)"
    echo "  ${GREEN}2)${NC} Check system status"
    echo "  ${GREEN}3)${NC} Check disk quota"
    echo "  ${GREEN}4)${NC} Clean up disk space"
    echo "  ${GREEN}5)${NC} Configure/verify CUDA"
    echo "  ${GREEN}6)${NC} Verify JAX GPU access"
    echo "  ${GREEN}7)${NC} Update data directory path"
    echo "  ${GREEN}8)${NC} Show experiment run instructions"
    echo "  ${GREEN}9)${NC} Exit"
    echo ""
    read -p "Choose option (1-9): " option
    
    case $option in
        1)
            run_full_setup
            ;;
        2)
            show_status
            ;;
        3)
            check_quota
            ;;
        4)
            cleanup_space
            ;;
        5)
            configure_cuda
            ;;
        6)
            source venv/bin/activate 2>/dev/null || print_warning "Virtual environment not found"
            verify_jax_cuda
            ;;
        7)
            check_data
            ;;
        8)
            show_run_instructions
            ;;
        9)
            print_info "Exiting"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

# Script entry point
check_directory

# If no arguments, show menu
if [ $# -eq 0 ]; then
    main_menu
else
    # Parse command line arguments
    case "$1" in
        setup)
            run_full_setup
            ;;
        status)
            show_status
            ;;
        quota)
            check_quota
            ;;
        clean)
            cleanup_space
            ;;
        cuda)
            configure_cuda
            verify_jax_cuda
            ;;
        help|--help|-h)
            echo "Usage: $0 [setup|status|quota|clean|cuda|help]"
            echo ""
            echo "Commands:"
            echo "  setup   - Run full environment setup"
            echo "  status  - Check system and experiment status"
            echo "  quota   - Check disk quota"
            echo "  clean   - Clean up disk space"
            echo "  cuda    - Configure and verify CUDA"
            echo "  help    - Show this help message"
            echo ""
            echo "Run without arguments for interactive menu"
            ;;
        *)
            echo "Unknown command: $1"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
fi

echo ""
print_status "Done!"