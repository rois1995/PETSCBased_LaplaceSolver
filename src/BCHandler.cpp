#include "BCHandler.h"
#include <petscdmplex.h>
#include <iostream>

BCHandler::BCHandler(Mesh &mesh, Assembler &assembler) : mesh_(mesh), assembler_(assembler) {}
BCHandler::~BCHandler() {}

PetscErrorCode BCHandler::applyNeumann() {
  // Placeholder: user should implement face quadrature here.
  PetscPrintf(mesh_.comm(), "applyNeumann(): not implemented - placeholder (no Neumann contribution applied)\n");
  return 0;
}

PetscErrorCode BCHandler::applyDirichlet(bool &hadAnyDirichlet) {
  PetscErrorCode ierr;
  DM dm = mesh_.getDM();
  Mat A = assembler_.getA();
  Vec b = assembler_.getB();

  DMLabel label = NULL;
  ierr = DMGetLabel(dm, "marker", &label); CHKERRQ(ierr);
  if (!label) DMGetLabel(dm, "Face Sets", &label);
  if (!label) DMGetLabel(dm, "boundary", &label);

  if (!label) {
    PetscPrintf(mesh_.comm(), "applyDirichlet(): no boundary label found (assuming no Dirichlet BCs)\n");
    hadAnyDirichlet = false;
    return 0;
  }

  IS isPoints = NULL;
  ierr = DMLabelGetStratumIS(label, 1, &isPoints); CHKERRQ(ierr); // demo: use label==1 as Dirichlet
  if (!isPoints) {
    PetscPrintf(mesh_.comm(), "applyDirichlet(): label exists but no points with value==1\n");
    hadAnyDirichlet = false;
    return 0;
  }

  const PetscInt *pts = nullptr; PetscInt nPoints = 0;
  ierr = ISGetLocalSize(isPoints, &nPoints); CHKERRQ(ierr);
  ierr = ISGetIndices(isPoints, &pts); CHKERRQ(ierr);

  PetscSection section; ierr = DMGetDefaultSection(dm, &section); CHKERRQ(ierr);
  PetscSection globalSection = nullptr; ierr = DMGetGlobalSection(dm, &globalSection); CHKERRQ(ierr);
  bool useGlobal = (globalSection != nullptr);

  std::vector<PetscInt> globals;
  globals.reserve(nPoints);
  for (PetscInt i=0;i<nPoints;++i) {
    PetscInt p = pts[i];
    PetscInt dof; ierr = PetscSectionGetDof(section, p, &dof); CHKERRQ(ierr);
    if (dof <= 0) continue;
    PetscInt off;
    if (useGlobal) ierr = PetscSectionGetOffset(globalSection, p, &off); CHKERRQ(ierr);
    else ierr = PetscSectionGetOffset(section, p, &off); CHKERRQ(ierr);
    globals.push_back(off);
  }
  ierr = ISRestoreIndices(isPoints, &pts); CHKERRQ(ierr);

  if (globals.empty()) {
    PetscPrintf(mesh_.comm(), "applyDirichlet(): no DOF vertices in label==1\n");
    ISDestroy(&isPoints);
    hadAnyDirichlet = false;
    return 0;
  }

  IS isDir; ierr = ISCreateGeneral(mesh_.comm(), (PetscInt)globals.size(), globals.data(), PETSC_COPY_VALUES, &isDir); CHKERRQ(ierr);

  PetscScalar dirVal = 0.0;
  ierr = MatZeroRowsIS(A, isDir, 1.0, NULL, NULL); CHKERRQ(ierr);
  for (auto g : globals) ierr = VecSetValue(b, g, dirVal, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  PetscPrintf(mesh_.comm(), "applyDirichlet(): applied Dirichlet to %zu DOFs (value=%g)\n", globals.size(), (double)dirVal);

  ISDestroy(&isDir);
  ISDestroy(&isPoints);
  hadAnyDirichlet = true;
  return 0;
}
