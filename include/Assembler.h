#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include "Mesh.h"
#include <petsc.h>
#include <vector>

class Assembler {
public:
  explicit Assembler(Mesh &mesh);
  ~Assembler();

  PetscErrorCode createSystem();               // create Mat, Vecs
  PetscErrorCode assembleSystem();             // assemble A and b
  Mat getA() const;
  Vec getB() const;
  Vec getX() const;

private:
  Mesh &mesh_;
  Mat A_;
  Vec b_;
  Vec x_;

  // cell assembly for simple linear tri/tet demo
  PetscErrorCode assembleCellElement(PetscInt spaceDim, PetscInt nv, const PetscReal *coords, PetscScalar *Ke, PetscScalar *fe);
};

#endif // ASSEMBLER_H
