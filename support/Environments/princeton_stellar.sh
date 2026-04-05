#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_load_modules() {
    module purge

    # 1. Compiler & MPI
    module load gcc-toolset/13
    module load openmpi/gcc/4.1.6

    # 2. HDF5 (Built against the exact GCC 10 and OpenMPI versions above)
    module load hdf5/gcc/openmpi-4.1.6/1.14.4

    # 3. Build System
    module load cmake/3.19.7

    # 4. Math & Core Libraries
    module load openblas/0.3.x
    module load gsl/2.6

    # 5. Python Environment
    # module load anaconda3/2024.2
}

spectre_load_runtime_env() {
    module load openmpi/gcc/4.1.6
    module load hdf5/gcc/openmpi-4.1.6/1.14.4
    module load openblas/0.3.x
    module load gsl/2.6
    export LD_LIBRARY_PATH="$HOME/libs/boost/lib:$LD_LIBRARY_PATH"
}

spectre_run_cmake() {
    if [ -z "${SPECTRE_HOME}" ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=/home/yk8311/libs/charm/mpi-linux-x86_64-smp \
        -D MEMORY_ALLOCATOR=SYSTEM \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_Fortran_COMPILER=gfortran \
        -D ENABLE_PYTHON=OFF \
        -D BUILD_PYTHON_BINDINGS=OFF \
        -D SPECTRE_FETCH_MISSING_DEPS=ON \
        -D CMAKE_SKIP_RPATH=ON \
        -D BOOST_ROOT=/home/yk8311/libs/boost \
        -D Python_EXECUTABLE=/usr/licensed/anaconda3/2024.2/bin/python3 \
        "$@" \
        "${SPECTRE_HOME}"
}
