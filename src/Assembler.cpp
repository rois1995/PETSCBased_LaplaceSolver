#include "Assembler.h"
#include <petscdmplex.h>
#include <petscsys.h>
#include <cassert>

Assembler::Assembler(Mesh &mesh) : mesh_(mesh), A_(nullptr), b_(nullptr), x_(nullptr) {}

Assembler::~Assembler() {
  if (A_) MatDestroy(&A_);
  if (b_) VecDestroy(&b_);
  if (x_) VecDestroy(&x_);
}

PetscErrorCode Assembler::createSystem() {
  PetscErrorCode ierr;
  DM dm = mesh_.getDM();
  ierr = DMCreateMatrix(dm, &A_); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &x_); CHKERRQ(ierr);
  ierr = VecDuplicate(x_, &b_); CHKERRQ(ierr);
  ierr = MatZeroEntries(A_); CHKERRQ(ierr);
  ierr = VecSet(b_, 0.0); CHKERRQ(ierr);
  return 0;
}

Mat Assembler::getA() const { return A_; }
Vec Assembler::getB() const { return b_; }
Vec Assembler::getX() const { return x_; }

PetscErrorCode Assembler::assembleSystem() {
  PetscErrorCode ierr;
  DM dm = mesh_.getDM();
  Vec coords;
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  if (!coords) {
    ierr = DMGetCoordinates(dm, &coords); CHKERRQ(ierr);
  }
  if (!coords) SETERRQ(mesh_.comm(), PETSC_ERR_ARG_WRONGSTATE, "Mesh has no coordinates");

  PetscInt cStart, cEnd; ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  PetscSection section; ierr = DMGetDefaultSection(dm, &section); CHKERRQ(ierr);
  PetscSection globalSection = nullptr; ierr = DMGetGlobalSection(dm, &globalSection); CHKERRQ(ierr);
  bool useGlobal = (globalSection != nullptr);

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt numCoords = 0; const PetscScalar *coordsArr = nullptr;
    ierr = DMPlexVecGetClosure(dm, NULL, coords, c, &numCoords, &coordsArr); CHKERRQ(ierr);
    if (numCoords <= 0) {
      DMPlexRestoreVecClosure(dm, NULL, coords, c, &numCoords, &coordsArr); CHKERRQ(ierr);
      continue;
    }
    PetscInt spaceDim; ierr = DMGetDimension(dm, &spaceDim); CHKERRQ(ierr);
    std::vector<PetscReal> coordBuf(numCoords);
    for (PetscInt i = 0; i < numCoords; ++i) coordBuf[i] = PetscRealPart(coordsArr[i]);

    PetscInt closureSize = 0; PetscInt *closure = nullptr;
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
    std::vector<PetscInt> vertexPoints;
    for (PetscInt i = 0; i < closureSize; ++i) {
      PetscInt p = closure[2*i];
      PetscInt dof; ierr = PetscSectionGetDof(section, p, &dof); CHKERRQ(ierr);
      if (dof > 0) vertexPoints.push_back(p);
    }

    PetscInt nv = (PetscInt)vertexPoints.size();
    std::vector<PetscScalar> Ke(nv*nv, 0.0);
    std::vector<PetscScalar> fe(nv, 0.0);

    ierr = assembleCellElement(spaceDim, nv, coordBuf.data(), Ke.data(), fe.data()); CHKERRQ(ierr);

    std::vector<PetscInt> globIdx(nv);
    for (PetscInt i = 0; i < nv; ++i) {
      PetscInt off;
      if (useGlobal) {
        ierr = PetscSectionGetOffset(globalSection, vertexPoints[i], &off); CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetOffset(section, vertexPoints[i], &off); CHKERRQ(ierr);
      }
      globIdx[i] = off;
    }

    for (PetscInt i = 0; i < nv; ++i) {
      for (PetscInt j = 0; j < nv; ++j) {
        ierr = MatSetValue(A_, globIdx[i], globIdx[j], Ke[i*nv + j], ADD_VALUES); CHKERRQ(ierr);
      }
      ierr = VecSetValue(b_, globIdx[i], fe[i], ADD_VALUES); CHKERRQ(ierr);
    }

    DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
    DMPlexRestoreVecClosure(dm, NULL, coords, c, &numCoords, &coordsArr); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b_); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b_); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode Assembler::assembleCellElement(PetscInt spaceDim, PetscInt nv, const PetscReal *coords, PetscScalar *Ke, PetscScalar *fe) {
  // Triangles (2D) and tetrahedra (3D) demonstration only
  if (spaceDim == 2 && nv == 3) {
    PetscReal x0 = coords[0], y0 = coords[1];
    PetscReal x1 = coords[2], y1 = coords[3];
    PetscReal x2 = coords[4], y2 = coords[5];
    PetscReal detJ = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
    PetscReal area = 0.5 * PetscAbsReal(detJ);
    if (area <= 0) SETERRQ(mesh_.comm(), PETSC_ERR_ARG_WRONG, "Degenerate triangle");
    PetscReal bx[3], by[3];
    bx[0] = (y1 - y2) / detJ;  by[0] = (x2 - x1) / detJ;
    bx[1] = (y2 - y0) / detJ;  by[1] = (x0 - x2) / detJ;
    bx[2] = (y0 - y1) / detJ;  by[2] = (x1 - x0) / detJ;
    for (PetscInt i = 0; i < 3; ++i)
      for (PetscInt j = 0; j < 3; ++j)
        Ke[i*3 + j] += (bx[i]*bx[j] + by[i]*by[j]) * area;
    PetscReal f = 1.0;
    for (PetscInt i = 0; i < 3; ++i) fe[i] += f * area / 3.0;
    return 0;
  } else if (spaceDim == 3 && nv == 4) {
    PetscReal x0 = coords[0], y0 = coords[1], z0 = coords[2];
    PetscReal x1 = coords[3], y1 = coords[4], z1 = coords[5];
    PetscReal x2 = coords[6], y2 = coords[7], z2 = coords[8];
    PetscReal x3 = coords[9], y3 = coords[10], z3 = coords[11];
    PetscReal M[9] = { x1-x0, x2-x0, x3-x0,
                       y1-y0, y2-y0, y3-y0,
                       z1-z0, z2-z0, z3-z0 };
    PetscReal detM = M[0]*(M[4]*M[8]-M[5]*M[7])
                   - M[1]*(M[3]*M[8]-M[5]*M[6])
                   + M[2]*(M[3]*M[7]-M[4]*M[6]);
    PetscReal vol = PetscAbsReal(detM)/6.0;
    if (vol <= 0) SETERRQ(mesh_.comm(), PETSC_ERR_ARG_WRONG, "Degenerate tet");
    PetscReal invM[9];
    invM[0] =  (M[4]*M[8] - M[5]*M[7]) / detM;
    invM[1] = -(M[1]*M[8] - M[2]*M[7]) / detM;
    invM[2] =  (M[1]*M[5] - M[2]*M[4]) / detM;
    invM[3] = -(M[3]*M[8] - M[5]*M[6]) / detM;
    invM[4] =  (M[0]*M[8] - M[2]*M[6]) / detM;
    invM[5] = -(M[0]*M[5] - M[2]*M[3]) / detM;
    invM[6] =  (M[3]*M[7] - M[4]*M[6]) / detM;
    invM[7] = -(M[0]*M[7] - M[1]*M[6]) / detM;
    invM[8] =  (M[0]*M[4] - M[1]*M[3]) / detM;
    PetscReal gx[4], gy[4], gz[4];
    for (PetscInt i=1;i<=3;++i) {
      gx[i] = invM[0*3 + (i-1)];
      gy[i] = invM[1*3 + (i-1)];
      gz[i] = invM[2*3 + (i-1)];
    }
    gx[0] = -(gx[1]+gx[2]+gx[3]);
    gy[0] = -(gy[1]+gy[2]+gy[3]);
    gz[0] = -(gz[1]+gz[2]+gz[3]);
    for (PetscInt i=0;i<4;++i)
      for (PetscInt j=0;j<4;++j)
        Ke[i*4 + j] += (gx[i]*gx[j] + gy[i]*gy[j] + gz[i]*gz[j]) * vol;
    PetscReal f=1.0;
    for (PetscInt i=0;i<4;++i) fe[i] += f * vol / 4.0;
    return 0;
  }
  SETERRQ(mesh_.comm(), PETSC_ERR_ARG_WRONG, "Element type not supported by Assembler demo");
  return 0;
}
