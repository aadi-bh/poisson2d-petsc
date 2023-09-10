/*
 * Solves 2D Poisson equation -∆u = f in Ω = [0,1]x[0,1]
 *                              u = g in ∂Ω
 *             using Jacobi iterationsh
 */
#include <cstddef>
#include <mpi.h>
#include <petsc.h>

PetscReal g(PetscInt, PetscInt) { return 0; }

PetscReal f(PetscInt, PetscInt) { return 1.; }

int main(int argc, char *argv[]) {
  char help[] = "Solves -∆u = f\n";
  // Number of points along both directions.
  PetscInt N = 4, nx, ny;
  // TODO Make an option for this too.
  const PetscReal eps = 1e-5;
  PetscReal h;
  constexpr PetscReal xmin = 0;
  constexpr PetscReal xmax = 1;
  constexpr PetscReal ymin = 0;
  constexpr PetscReal ymax = 1;

  // TODO Error-checking
  // da stands for "distributed array"
  DM da;
  Vec u_global;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  
  // TODO These are macros, and don't return anything, so I can't use with PetscCall
  PetscOptionsBegin(PETSC_COMM_WORLD, "size_", "options for number of grid of points", "");

  PetscCall(PetscOptionsInt("-N", "size of grid along each axis", "main.cc", N,
                            &N, NULL));
  // TODO no more parameters?
  PetscOptionsEnd();

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
               /* boundary types */ DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
               /* stencil shape */ DMDA_STENCIL_STAR,
               /* grid size */ N, N,
               /* ranks in each dim */ PETSC_DECIDE, PETSC_DECIDE,
               /* dofs per node, stencil width */ 1, 1,
               /* TODO what are lx and ly? */ NULL, NULL,
               /* DM* */ &da));

  // Sets parameters from the options database
  PetscCall(DMSetFromOptions(da));
  // Now actual set_up_. That was just set params from options.
  PetscCall(DMSetUp(da));
  // Sets uniform coordinates for the grid. z-values are ignored in 2D
  PetscCall(DMDASetUniformCoordinates(da, xmin, xmax, ymin, ymax, 0., 0.));

  // Find h
  PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
              NULL, NULL, NULL));
  if (nx != ny)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "nx not equal to ny\n"));
  PetscReal dx = (xmax - xmin) / (PetscReal)(nx);
  PetscReal dy = (ymax - ymin) / (PetscReal)(ny);
  h = dx;

  PetscCall(DMCreateGlobalVector(da, &u_global));
  PetscCall(PetscObjectSetName((PetscObject)u_global, "Solution"));

  PetscScalar **u;
  Vec u_local;
  PetscInt ibeg, jbeg, nlocx, nlocy;
  PetscCall(DMDAGetCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL));
  PetscCall(DMDAVecGetArray(da, u_global, &u));

  // set the initial guess to zero, and boundary cells to g.
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocy; ++i) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        // Then this point is on the boundary! Set it to g(x,y)
        PetscReal x = xmin + i * dx;
        PetscReal y = ymin + i * dy;
        u[j][i] = g(x, y);
      }
      u[j][i] = 0.;
    }
  PetscCall(DMDAVecRestoreArray(da, u_global, &u));
  // TODO save this initial condition please

  // Creates a vector for temporary use.
  // It has spaces for the local ghosts too,
  // but none of the values are initialised, so expect garbage.
  PetscCall(DMGetLocalVector(da, &u_local));

  // suffix l for local
  PetscInt il, jl, nl, ml;
  // This gives us the bottom-left corner of the local vector,
  // including the ghost indices for this rank.
  // nl, ml are the dimensions, also including the ghosts.
  PetscCall(DMDAGetGhostCorners(da, &il, &jl, NULL, &nl, &ml, NULL));

  PetscCall(DMGlobalToLocalBegin(da, u_global, INSERT_VALUES, u_local));
  PetscCall(DMGlobalToLocalEnd(da, u_global, INSERT_VALUES, u_local));

  // Now fill in the array u with the local data
  PetscCall(DMDAVecGetArrayRead(da, u_local, &u));

  // Place to store the next iterate
  PetscScalar **u_new;
  // Since we are going to be filling the value of u_new,
  // we don't need to initialise it. Hence no GetRead, just Get
  PetscCall(DMDAVecGetArray(da, u_global, &u_new));
  PetscPrintf(PETSC_COMM_WORLD, "%d\n", u_global == &u_new);
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocx; ++i) {
      PetscReal x = xmin + i * dx;
      PetscReal y = ymin + i * dy;
      u_new[j][i] = 0.25 * (h * h * f(x, y) - u[j - 1][i] + u[j + 1][i] +
                            u[j][i - 1] + u[j][i + 1]);
    }

  // TODO need better comments
  // Put u's values in the u_local array
  PetscCall(DMDAVecRestoreArrayRead(da, u_local, &u));
  // Put u_new's values into the u_global array
  PetscCall(DMDAVecRestoreArray(da, u_global, u_new));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Completed one sweep\n"));

  // TODO Next line causes a segfault. Removing the VecRestoreArray(da, u_g, u_new)
  // works somehow.
  // PetscCall(VecDestroy(&u_global));
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}
