#include "Mesh.h"
#include "Assembler.h"
#include "BCHandler.h"
#include "PoissonSolver.h"
#include <petscsys.h>
#include <iostream>

int main(int argc, char **argv) {
  PetscInitialize(&argc, &argv, NULL, "Poisson node-centered PETSc DMPlex example (multi-file)");

  MPI_Comm comm = PETSC_COMM_WORLD;
  int rank; MPI_Comm_rank(comm, &rank);

  if (argc < 2) {
    if (rank == 0) std::cerr << "Usage: " << argv[0] << " mesh.cgns [output.vtu]\n";
    PetscFinalize();
    return 1;
  }
  std::string meshfile = argv[1];
  std::string outvtk = "solution.vtk";
  if (argc >= 3) outvtk = argv[2];

  Mesh mesh(comm);
  CHECK(mesh.loadFromCGNS(meshfile));
  CHECK(mesh.distribute());
  CHECK(mesh.setupNodeCenteredSection());

  Assembler assembler(mesh);
  BCHandler bc(mesh, assembler);
  PoissonSolver solver(mesh, assembler, bc);

  CHECK(solver.run());

  // Write vtk
  CHECK(solver.writeVTK(outvtk));

  PetscFinalize();
  return 0;
}
