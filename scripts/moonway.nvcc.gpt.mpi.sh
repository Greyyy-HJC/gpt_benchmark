#!/bin/bash
#
# Get compilers
#

echo "Activating Conda environment: pygpt"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygpt 

# Environment variables for MPI and UCX
export OMPI_MCA_btl=^uct,openib
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_SCHEME=put_zcopy

# Ensure the correct CUDA paths are set
export PATH=$CONDA_PREFIX/bin:$CONDA_PREFIX/cuda/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/cuda/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Compiler settings
export CXX=$CONDA_PREFIX/bin/g++
export MPICXX=$CONDA_PREFIX/bin/mpicxx

echo "Using g++: $(which g++)"
echo "Using nvcc: $(which nvcc)"
echo "Using mpicxx: $(which mpicxx)"

#
# Get root directory
#
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

#
# Precompile Python code
#
echo "Compiling gpt Python modules"
python -m compileall ${root}/lib/gpt || { echo "Python compilation failed"; exit 1; }

#
# Create dependencies and download
#
dep=${root}/dependencies
if [ ! -f ${dep}/Grid/build/Grid/libGrid.a ]; then

    if [ -d ${dep} ]; then
        echo "$dep already exists; rm -rf $dep before bootstrapping again"
        exit 1
    fi

    mkdir -p ${dep}
    cd ${dep}

    #
    # Lime
    #
    echo "Installing Lime..."
    wget https://github.com/usqcd-software/c-lime/tarball/master
    tar xzf master
    mv usqcd-software-c-lime* lime
    rm -f master
    cd lime
    ./autogen.sh
    ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC"
    make || { echo "Lime build failed"; exit 1; }
    cd ..

    #
    # Grid
    #
    echo "Installing Grid..."
    git clone https://github.com/dbollweg/Grid.git
    cd Grid
    git checkout gpt_proton
    ./bootstrap.sh
    mkdir build
    cd build
    ../configure --enable-comms=mpi-auto \
        --enable-unified=no \
        --enable-accelerator=cuda \
        --enable-alloc-align=4k \
        --enable-shm=nvlink \
        --enable-simd=GPU \
        --with-fftw=$CONDA_PREFIX \
        --with-lime=${dep}/lime \
        CXX="$CONDA_PREFIX/cuda/usr/local/cuda-12.1/bin/nvcc" \
        MPICXX="$CONDA_PREFIX/bin/mpicxx" \
        CXXFLAGS="-ccbin $CONDA_PREFIX/bin/g++ -gencode arch=compute_86,code=sm_86 -std=c++17 --cudart shared -lcublas -Xcompiler=-fPIC,-O2" \
        LIBS="-lrt" \
        LDFLAGS="--cudart shared -L$CONDA_PREFIX/lib -lmpi -lcublas" || { echo "Grid configure failed"; exit 1; }

    cd Grid
    make -j 8 || { echo "Grid make failed"; exit 1; }
fi

if [ ! -f ${root}/lib/cgpt/build/cgpt.so ]; then
    #
    # cgpt
    #
    echo "Installing CGPT..."
    cd ${root}/lib/cgpt
    ./make ${dep}/Grid/build 8 || { echo "cgpt make failed"; exit 1; }
fi

echo "Installation complete!"
echo "To use:"
echo "source ${root}/lib/cgpt/build/source.sh"