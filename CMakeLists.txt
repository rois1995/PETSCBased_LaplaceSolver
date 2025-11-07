cmake_minimum_required(VERSION 3.13)
project(PoissonPetScNodeCentered LANGUAGES CXX)

set(PETSC_DIR "/home/rausa/Software/petsc/build")
set(PETSC_ARCH "arch-linux-c-debug")

set(PETSC_INCLUDE_DIRS
    ${PETSC_DIR}/include
)

set(PETSC_LIBRARIES
    ${PETSC_DIR}/lib/libpetsc.so
)

link_directories(${PETSC_DIR}/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PETSC_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

file(GLOB SRCFILES src/*.cpp)

add_executable(poisson
  ${SRCFILES}
  Main.cpp
)

# link PETSc
target_link_libraries(poisson PRIVATE ${PETSC_LIBRARIES})
# propagate PETSc compile flags
if(PETSC_CFLAGS)
  string(REPLACE ";" " " PETSC_CFLAGS_STR "${PETSC_CFLAGS}")
  set_target_properties(poisson PROPERTIES COMPILE_FLAGS "${PETSC_CFLAGS_STR}")
endif()

# Install target (optional)
install(TARGETS poisson DESTINATION bin)