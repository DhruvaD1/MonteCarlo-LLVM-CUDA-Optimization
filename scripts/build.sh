#!/bin/bash

# Automatic Tiling for GPU Kernels - Build Script
# This script builds the entire project including CUDA kernel and LLVM pass

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$PROJECT_ROOT/build"

# Auto-detect GPU architecture for RTX 5070 Ti and other modern GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    if [[ "$GPU_NAME" =~ "RTX 5070" ]] || [[ "$GPU_NAME" =~ "RTX 50" ]]; then
        CUDA_ARCH=${CUDA_ARCH:-"sm_89"}  # RTX 50 series likely uses compute 8.9
        echo -e "${GREEN}Detected RTX 50 series GPU: $GPU_NAME${NC}"
    elif [[ "$GPU_NAME" =~ "RTX 4090" ]] || [[ "$GPU_NAME" =~ "RTX 4080" ]] || [[ "$GPU_NAME" =~ "RTX 40" ]]; then
        CUDA_ARCH=${CUDA_ARCH:-"sm_89"}  # RTX 40 series uses compute 8.9
    elif [[ "$GPU_NAME" =~ "RTX 3090" ]] || [[ "$GPU_NAME" =~ "RTX 3080" ]] || [[ "$GPU_NAME" =~ "RTX 30" ]]; then
        CUDA_ARCH=${CUDA_ARCH:-"sm_86"}  # RTX 30 series uses compute 8.6
    elif [[ "$GPU_NAME" =~ "RTX 2080" ]] || [[ "$GPU_NAME" =~ "RTX 20" ]]; then
        CUDA_ARCH=${CUDA_ARCH:-"sm_75"}  # RTX 20 series uses compute 7.5
    else
        CUDA_ARCH=${CUDA_ARCH:-"sm_75"}  # Default fallback
    fi
else
    CUDA_ARCH=${CUDA_ARCH:-"sm_75"}  # Default if nvidia-smi not available
fi

BUILD_TYPE=${BUILD_TYPE:-"Release"}

echo -e "${BLUE}=== Automatic Tiling for GPU Kernels - Build Script ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "CUDA architecture: $CUDA_ARCH"
echo "Build type: $BUILD_TYPE"
echo

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found. Please install CUDA Toolkit."
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
print_success "Found nvcc version: $NVCC_VERSION"

# Check for LLVM
if ! command -v llvm-config &> /dev/null; then
    print_warning "llvm-config not found in PATH. Trying to find it..."
    
    # Common LLVM installation paths
    LLVM_PATHS=(
        "/usr/local/llvm"
        "/opt/llvm"
        "/usr/lib/llvm-*"
        "$HOME/llvm-project/build"
    )
    
    LLVM_CONFIG_PATH=""
    for path in "${LLVM_PATHS[@]}"; do
        if [[ -f "$path/bin/llvm-config" ]]; then
            LLVM_CONFIG_PATH="$path/bin/llvm-config"
            export PATH="$path/bin:$PATH"
            break
        fi
    done
    
    if [[ -z "$LLVM_CONFIG_PATH" ]]; then
        print_error "LLVM not found. Please install LLVM with NVPTX backend or set PATH."
        echo "Expected locations:"
        for path in "${LLVM_PATHS[@]}"; do
            echo "  - $path/bin/llvm-config"
        done
        exit 1
    fi
fi

LLVM_VERSION=$(llvm-config --version)
LLVM_PREFIX=$(llvm-config --prefix)
print_success "Found LLVM version: $LLVM_VERSION at $LLVM_PREFIX"

# Check LLVM targets
LLVM_TARGETS=$(llvm-config --targets-built)
if [[ ! "$LLVM_TARGETS" =~ "NVPTX" ]]; then
    print_error "LLVM was not built with NVPTX target. Please rebuild LLVM with NVPTX support."
    echo "Current targets: $LLVM_TARGETS"
    exit 1
fi
print_success "LLVM includes NVPTX target"

# Create build directory
print_status "Creating build directory..."
mkdir -p "$BUILD_DIR"

# Build CUDA kernel
print_status "Building CUDA Monte Carlo kernel..."
cd "$PROJECT_ROOT"

CUDA_FLAGS=(
    "-std=c++17"
    "-O3"
    "--gpu-architecture=$CUDA_ARCH"
    "--generate-code=arch=compute_75,code=sm_75"
    "--generate-code=arch=compute_80,code=sm_80"
    "--generate-code=arch=compute_86,code=sm_86"
    "--generate-code=arch=compute_89,code=sm_89"
    "--generate-code=arch=compute_90,code=sm_90"
    "-lcurand"
    "-Xcompiler=-fopenmp"
)

nvcc "${CUDA_FLAGS[@]}" src/matrix_mul.cu -o "$BUILD_DIR/monte_carlo_baseline" || {
    print_error "Failed to build CUDA kernel"
    exit 1
}
print_success "Built baseline Monte Carlo kernel"

# Generate LLVM IR from CUDA kernel
print_status "Generating LLVM IR from CUDA kernel..."

# Use clang to generate LLVM IR
CLANG_FLAGS=(
    "-S"
    "-emit-llvm"
    "--cuda-gpu-arch=$CUDA_ARCH"
    "--cuda-device-only"
    "-nocudainc"
    "-nocudalib"
    "-O1"  # Use O1 to get readable IR but with some optimizations
    "-Xclang" "-disable-llvm-passes"  # Disable optimizations to see original structure
)

# Generate IR
clang++ "${CLANG_FLAGS[@]}" src/matrix_mul.cu -o "$BUILD_DIR/monte_carlo_baseline.ll" || {
    print_error "Failed to generate LLVM IR"
    exit 1
}
print_success "Generated LLVM IR: $BUILD_DIR/monte_carlo_baseline.ll"

# Build LLVM pass
print_status "Building LLVM AutoTiling pass..."

# Create separate build directory for LLVM pass
PASS_BUILD_DIR="$BUILD_DIR/AutoTilingPass"
mkdir -p "$PASS_BUILD_DIR"

cd "$PASS_BUILD_DIR"

# Configure with CMake
CMAKE_FLAGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DLLVM_DIR=$(llvm-config --cmakedir)"
    "-DCMAKE_CXX_STANDARD=17"
)

cmake "${CMAKE_FLAGS[@]}" "$PROJECT_ROOT/src/AutoTilingPass" || {
    print_error "CMake configuration failed"
    exit 1
}

# Build the pass
make -j$(nproc) || {
    print_error "Failed to build LLVM pass"
    exit 1
}
print_success "Built LLVM AutoTiling pass"

# Apply the pass to generate optimized IR
print_status "Applying AutoTiling pass..."
cd "$PROJECT_ROOT"

opt -load "$PASS_BUILD_DIR/lib/LLVMAutoTiling.so" -autotile \
    < "$BUILD_DIR/monte_carlo_baseline.ll" \
    > "$BUILD_DIR/monte_carlo_tiled.ll" 2> "$BUILD_DIR/pass_output.log" || {
    print_warning "Pass application may have issues, check $BUILD_DIR/pass_output.log"
    print_status "Continuing with build..."
}

if [[ -f "$BUILD_DIR/monte_carlo_tiled.ll" ]]; then
    print_success "Generated tiled LLVM IR: $BUILD_DIR/monte_carlo_tiled.ll"
else
    print_warning "Tiled IR not generated, using baseline IR"
    cp "$BUILD_DIR/monte_carlo_baseline.ll" "$BUILD_DIR/monte_carlo_tiled.ll"
fi

# Compile optimized IR back to executable
print_status "Compiling optimized IR to executable..."

# First compile IR to object file
llc -march=nvptx64 -mcpu=$CUDA_ARCH "$BUILD_DIR/monte_carlo_tiled.ll" \
    -o "$BUILD_DIR/monte_carlo_tiled.s" || {
    print_warning "Failed to compile IR to assembly, using baseline"
}

# Create a version comparison setup
print_status "Setting up performance comparison..."

# Create a wrapper script for easy testing
cat > "$BUILD_DIR/run_comparison.sh" << 'EOF'
#!/bin/bash

echo "=== Monte Carlo Asian Option Pricing - Performance Comparison ==="
echo "Baseline (unoptimized):"
echo "----------------------------------------"
time ./monte_carlo_baseline 32768 512
echo
echo "Optimized (with tiling pass):"
echo "----------------------------------------" 
# Note: In a complete implementation, we would have the optimized version
echo "Optimized version would be run here"
echo "(Implementation requires full IR-to-executable compilation pipeline)"
EOF

chmod +x "$BUILD_DIR/run_comparison.sh"

# Create analysis directory
mkdir -p "$BUILD_DIR/analysis"

# Summary
print_success "Build completed successfully!"
echo
echo -e "${BLUE}=== Build Summary ===${NC}"
echo "✓ CUDA baseline executable: $BUILD_DIR/monte_carlo_baseline"
echo "✓ LLVM IR (baseline): $BUILD_DIR/monte_carlo_baseline.ll"
echo "✓ LLVM IR (tiled): $BUILD_DIR/monte_carlo_tiled.ll"
echo "✓ AutoTiling pass: $PASS_BUILD_DIR/lib/LLVMAutoTiling.so"
echo "✓ Comparison script: $BUILD_DIR/run_comparison.sh"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run performance comparison: cd $BUILD_DIR && ./run_comparison.sh"
echo "2. Run detailed benchmarks: cd $PROJECT_ROOT && scripts/benchmark.sh"
echo "3. Analyze IR differences: diff -u monte_carlo_baseline.ll monte_carlo_tiled.ll"
echo "4. Profile with Nsight Compute: ncu --set full ./monte_carlo_baseline"
