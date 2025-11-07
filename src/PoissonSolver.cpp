#include "PoissonSolver.h"
#include <petscdmplex.h>
#include <petscviewer.h>

PoissonSolver::PoissonSolver(Mesh &mesh, Assembler &assembler, BCHandler &bc)
  : mesh_(mesh), assembler_(assembler), bc_(bc) {}

PoissonSolver::~PoissonSolver() {}

PetscErrorCode PoissonSolver::run() {
  PetscErrorCode ierr;
  ierr = assembler_.createSystem(); CHKERRQ(ierr);
  ierr = assembler_.assembleSystem(); CHKERRQ(ierr);

  bool hadDirichlet = false;
  ierr = bc_.applyNeumann(); CHKERRQ(ierr);
  ierr = bc_.applyDirichlet(hadDirichlet); CHKERRQ(ierr);

  // handle pure Neumann (singular stiffness)
  if (!hadDirichlet) {
    PetscPrintf(mesh_.comm(), "No Dirichlet BCs detected: attaching constant nullspace.\n");
    MatNullSpace nullsp;
    ierr = MatNullSpaceCreate(mesh_.comm(), PETSC_TRUE, 0, NULL, &nullsp); CHKERRQ(ierr);
    ierr = MatSetNullSpace(assembler_.getA(), nullsp); CHKERRQ(ierr);
    MatNullSpaceDestroy(&nullsp);
  }

  ierr = solveSystem(assembler_.getA(), assembler_.getB(), assembler_.getX()); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode PoissonSolver::solveSystem(Mat A, Vec b, Vec x) {
  PetscErrorCode ierr;
  KSP ksp;
  ierr = KSPCreate(mesh_.comm(), &ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

  KSPConvergedReason reason; ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
  PetscInt its; ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
  PetscPrintf(mesh_.comm(), "KSP finished with reason %d after %d iterations\n", reason, its);

  KSPDestroy(&ksp);
  return 0;
}

PetscErrorCode PoissonSolver::writeVTK(const std::string &filename) {
  PetscErrorCode ierr;
  DM dm = mesh_.getDM();
  Vec x = assembler_.getX();

  PetscViewer viewer;
  ierr = PetscViewerVTKOpen(mesh_.comm(), filename.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);

  // There are two practical ways people use:
  // 1) Use DMPlexVTKWriteAll to write the DM+attached fields that have been registered with the viewer.
  // 2) Directly VecView(x, viewer) (works if Vec has a vertex distribution).
  //
  // We'll attempt DMPlexVTKWriteAll (writes mesh + any fields provided to the viewer).
  // To make DMPlexVTKWriteAll pick up our vector field, we set a name and then call VecView.
  ierr = PetscObjectSetName((PetscObject)x, (char*)"solution"); CHKERRQ(ierr);
  ierr = VecView(x, viewer); CHKERRQ(ierr);

  // DMPlexVTKWriteAll writes all the fields collected by the VTK viewer for this DM.
  // It's also safe to call DMView(dm, viewer) but that usually prints only the mesh.
  ierr = DMPlexVTKWriteAll((PetscObject)dm, viewer); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscPrintf(mesh_.comm(), "Wrote VTK output to %s\n", filename.c_str());
  return 0;
}
