#ifndef POISSONSOLVER_H
#define POISSONSOLVER_H

#include "Mesh.h"
#include "Assembler.h"
#include "BCHandler.h"
#include <petsc.h>

class PoissonSolver {
public:
  PoissonSolver(Mesh &mesh, Assembler &assembler, BCHandler &bc);
  ~PoissonSolver();

  PetscErrorCode run();
  // Write VTK output: writes mesh + solution to filename (.vtu/.vtk)
  PetscErrorCode writeVTK(const std::string &filename);

private:
  Mesh &mesh_;
  Assembler &assembler_;
  BCHandler &bc_;
  PetscErrorCode solveSystem(Mat A, Vec b, Vec x);
};

#endif // POISSONSOLVER_H
