#ifndef MESH_H
#define MESH_H

#include <petscdmplex.h>
#include <string>

class Mesh {
public:
  explicit Mesh(MPI_Comm comm = PETSC_COMM_WORLD);
  ~Mesh();

  PetscErrorCode loadFromCGNS(const std::string &filename);
  PetscErrorCode distribute();
  PetscErrorCode setupNodeCenteredSection();

  DM getDM() const;
  MPI_Comm comm() const;
  PetscInt dim() const;

private:
  MPI_Comm comm_;
  DM dm_;
  PetscInt dim_;
};

#endif // MESH_H
