#include "Mesh.h"
#include <petscsys.h>
#include <iostream>

Mesh::Mesh(MPI_Comm comm) : comm_(comm), dm_(nullptr), dim_(0) {}

Mesh::~Mesh() {
  if (dm_) DMDestroy(&dm_);
}

PetscErrorCode Mesh::loadFromCGNS(const std::string &filename) {
  PetscErrorCode ierr;
  ierr = DMPlexCreateFromFile(comm_, filename.c_str(), PETSC_TRUE, &dm_); CHKERRQ(ierr);
  ierr = DMGetDimension(dm_, &dim_); CHKERRQ(ierr);
  PetscPrintf(comm_, "Mesh loaded: '%s', dim=%d\n", filename.c_str(), dim_);
  return 0;
}

PetscErrorCode Mesh::distribute() {
  PetscErrorCode ierr;
  DM dist = nullptr;
  ierr = DMPlexDistribute(dm_, 0, NULL, &dist); CHKERRQ(ierr);
  if (dist) {
    DMDestroy(&dm_);
    dm_ = dist;
    PetscPrintf(comm_, "Mesh distributed across ranks.\n");
  }
  return 0;
}

PetscErrorCode Mesh::setupNodeCenteredSection() {
  PetscErrorCode ierr;
  PetscInt vStart, vEnd;
  ierr = DMPlexGetDepthStratum(dm_, 0, &vStart, &vEnd); CHKERRQ(ierr);
  PetscSection section;
  ierr = PetscSectionCreate(comm_, &section); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, vStart, vEnd); CHKERRQ(ierr);
  for (PetscInt v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(section, v, 1); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(section, v, 0, 1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section); CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm_, section); CHKERRQ(ierr);
  PetscSectionDestroy(&section);
  return 0;
}

DM Mesh::getDM() const { return dm_; }
MPI_Comm Mesh::comm() const { return comm_; }
PetscInt Mesh::dim() const { return dim_; }
