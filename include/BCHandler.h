#ifndef BCHANDLER_H
#define BCHANDLER_H

#include "Mesh.h"
#include "Assembler.h"
#include <petsc.h>

class BCHandler {
public:
  BCHandler(Mesh &mesh, Assembler &assembler);
  ~BCHandler();

  PetscErrorCode applyNeumann();               // placeholder for face integration
  PetscErrorCode applyDirichlet(bool &hadAnyDirichlet); // strongly enforce Dirichlet via MatZeroRowsIS

private:
  Mesh &mesh_;
  Assembler &assembler_;
};

#endif // BCHANDLER_H
