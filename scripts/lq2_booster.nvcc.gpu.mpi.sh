#!/bin/bash
#
# Get compilers
#
module load cmake
module load openmpi/4.1.5_gcc_12.3.0
module load lapack
module load gcc/12.3.0
module load fftw/3.3.10_gcc_12.3.0
module load cuda/12.2.1

export OMPI_MCA_btl=^uct,openib
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_SCHEME=put_zcopy

pip3 install numpy --user

#
# Get root directory
#
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

#
# Precompile python
#
echo "Compile gpt"
python3 -m compileall ${root}/lib/gpt

#
# Create dependencies and download
#
dep=${root}/dependencies
if [ ! -f ${dep}/Grid/build/Grid/libGrid.a ];
then

        if [ -d ${dep} ];
        then
            echo "$dep already exists ; rm -rf $dep before bootstrapping again"
            exit 1
        fi

        mkdir -p ${dep}
        cd ${dep}

        #
        # Lime
        #
        wget https://github.com/usqcd-software/c-lime/tarball/master
        tar xzf master
        mv usqcd-software-c-lime* lime
        rm -f master
        cd lime
        ./autogen.sh
        ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC"
        make
        cd ..

        #
        # Grid
        #

        # Use EBROOTFFTW as FFTW root
        FFTW_ROOT=${EBROOTFFTW}

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
             --with-fftw=$FFTW_ROOT \
             --with-lime=${dep}/lime \
             CXX=nvcc \
             MPICXX=mpicxx \
             CXXFLAGS="-ccbin g++ -gencode arch=compute_80,code=sm_80 -std=c++17 --cudart shared -lcublas -Xcompiler=-fPIC" \
             LIBS="-lrt" \
             LDFLAGS="--cudart shared -L/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/lib -lmpi -lcublas"

        cd Grid
        make -j || { echo "Grid make failed"; exit 1; }
fi

if [ ! -f ${root}/lib/cgpt/build/cgpt.so ];
then
        #
        # cgpt
        #
        cd ${root}/lib/cgpt
        ./make ${dep}/Grid/build 16 || { echo "cgpt make failed"; exit 1; }
fi


echo "To use:"
echo "source ${root}/lib/cgpt/build/source.sh"




