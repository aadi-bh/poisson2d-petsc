/*
 * Solves 2D Poisson equation -∆u = f in Ω = [0,1]x[0,1]
 *                              u = g in ∂Ω
 *             using Jacobi iterationsh
 */
#include <cstddef>
#include <mpi.h>
#include <petsc.h>

PetscReal g(PetscInt x, PetscInt y)
{
  return 0;
}

PetscReal f(PetscInt x, PetscInt y)
{
  return 1.;
}

int main(int argc, char *argv[]) {
  char help[] = "Solves -∆u = f\n";
  // Number of points along both directions.
  PetscInt N = 4, nx, ny;
  PetscReal h;
  constexpr PetscReal xmin = 0;
  constexpr PetscReal xmax = 1;
  constexpr PetscReal ymin = 0;
  constexpr PetscReal ymax = 1;

  // TODO Error-checking
  // da stands for "distributed array"
  DM da;
  Vec u_global;

  PetscInitialize(&argc, &argv, NULL, help);

  PetscOptionsBegin(PETSC_COMM_WORLD, "size_",
                    "options for number of grid of points", "");
  PetscOptionsInt("-N", "size of grid along each axis", "main.cc", N, &N, NULL);
  // TODO no more parameters?
  PetscOptionsEnd();

  DMDACreate2d(PETSC_COMM_WORLD,
               /* boundary types */ DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
               /* stencil shape */ DMDA_STENCIL_STAR,
               /* grid size */ N, N,
               /* ranks in each dim */ PETSC_DECIDE, PETSC_DECIDE,
               /* dofs per node, stencil width */ 1, 1,
               /* TODO what are lx and ly? */ NULL, NULL,
               /* DM* */ &da);

  // Sets parameters from the options database
  DMSetFromOptions(da);
  // Now actual set_up_. That was just set params from options.
  DMSetUp(da);
  // Sets uniform coordinates for the grid. z-values are ignored in 2D
  DMDASetUniformCoordinates(da, xmin, xmax, ymin, ymax, 0., 0.);

  // Find h
  DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  if (nx != ny)
    PetscPrintf(PETSC_COMM_WORLD, "nx not equal to ny\n");
  PetscReal dx = (xmax-xmin)/(PetscReal) (nx);
  PetscReal dy = (ymax-ymin)/(PetscReal) (ny);
  h = dx;


  // Create the RHS vector.
  // We'll make the local one later, this holds the global rhs.
  DMCreateGlobalVector(da, &u_global);
  PetscObjectSetName((PetscObject) u_global, "Solution");

  PetscScalar **u;
  Vec u_local;
  PetscInt ibeg, jbeg, nlocx, nlocy;
  DMDAGetCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL);
  DMDAVecGetArray(da, u_global, &u);

  // set the initial guess to zero, and boundary cells to g.
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocy; ++i)
    {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
      {
        // Then this point is on the boundary! Set it to g(x,y)
        PetscReal x = xmin + i * dx;
        PetscReal y = ymin + i * dy;
        u[j][i] = g(x,y);
      }
      u[j][i] = 0.;
    }
  DMDAVecRestoreArray(da, u_global, &u);
  // TODO save this initial condition please
  
  // Creates a vector for temporary use.
  // It has spaces for the local ghosts too,
  // but none of the values are initialised, so expect garbage.
  DMGetLocalVector(da, &u_local);

  // suffix l for local
  PetscInt il, jl, nl, ml;
  // This gives us the bottom-left corner of the local vector,
  // including the ghost indices for this rank.
  // nl, ml are the dimensions, also including the ghosts.
  DMDAGetGhostCorners(da, &il, &jl, NULL, &nl, &ml, NULL);

  DMGlobalToLocalBegin(da, u_global, INSERT_VALUES, u_local);
  DMGlobalToLocalEnd(da, u_global, INSERT_VALUES, u_local);

  // Now fill in the array u with the local data
  DMDAVecGetArrayRead(da, u_local, &u);

  // Place to store the next iterate
  PetscScalar **u_new;
  // Since we are going to be filling the value of u_new, 
  // we don't need to initialise it. Hence no GetRead, just Get
  DMDAVecGetArray(da, u_global, &u_new);

  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocx; ++i)
    {
      PetscReal x = xmin + i * dx;
      PetscReal y = ymin + i * dy;
      u_new[j][i] = 0.25 * (h*h * f(x,y) - u[j-1][i] + u[j+1][i]
                            +u[j][i-1] + u[j][i+1 ]);
    }

  // TODO need better comments
  // Put u's values in the u_local array
  DMDAVecRestoreArrayRead(da, u_local, &u);
  // Put u_new's values into the u_global array
  DMDAVecRestoreArray(da, u_global, u_new);

  PetscPrintf(PETSC_COMM_WORLD, "Completed one sweep\n");

  VecDestroy(&u_global);
  DMRestoreLocalVector(da, &u_local);
  DMDestroy(&da);

  PetscFinalize();
  return 0;
}
