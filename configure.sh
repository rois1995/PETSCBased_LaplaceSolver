#!/bin/bash
PETSC_DIR=/home/rausa/Software/petsc
PETSC_ARCH=arch-linux-c-debug

if [[ -d build ]]
then
  rm -rf build
fi

mkdir -p build

cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=${PETSC_DIR}/build/bin/mpicxx \
  -DPETSC_DIR=${PETSC_DIR} \
  -DPETSC_ARCH=${PETSC_ARCH}

cmake --build build
