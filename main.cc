/*
 * Solves 2D Poisson equation -∆u = f = rhs            in Ω = [0,1]x[0,1]
 *                              u = g = boundary_value in ∂Ω
 *             using Jacobi iterations.
 */
#include <cstddef>
#include <mpi.h>
#include <petsc.h>

PetscReal rhs(PetscInt x, PetscInt y) { return 4. * (pow(x, 4) + pow(y, 4)); }

PetscReal boundary_value(PetscReal x, PetscReal y) { return x * x + y * y; }

int main(int argc, char *argv[]) {
  char help[] = "Solves -∆u = f\n";
  // Number of points along both directions.
  PetscInt N = 4, nx, ny;
  // TODO Make an option for these too.
  const PetscReal eps = 1e-5;
  const PetscInt itermax = 10;
  PetscReal h;
  constexpr PetscReal xmin = -1.;
  constexpr PetscReal xmax = 1.;
  constexpr PetscReal ymin = -1.;
  constexpr PetscReal ymax = 1;

  // TODO Error-checking
  // da stands for "distributed array"
  DM da;
  Vec u_local, u_global;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // TODO These are macros, and don't return anything, so I can't use with
  // PetscCall
  PetscOptionsBegin(PETSC_COMM_WORLD, "size_",
                    "options for number of grid of points", "");

  PetscCall(PetscOptionsInt("-N", "size of grid along each axis", "main.cc", N,
                            &N, NULL));
  // TODO no more parameters?
  PetscOptionsEnd();
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                         /* boundary types */ DM_BOUNDARY_GHOSTED,
                         DM_BOUNDARY_GHOSTED,
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
  PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL));
  if (nx != ny)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "nx not equal to ny\n"));
  PetscReal dx = (xmax - xmin) / (PetscReal)(nx);
  PetscReal dy = (ymax - ymin) / (PetscReal)(ny);
  h = dx;

  PetscCall(DMCreateLocalVector(da, &u_local));
  PetscCall(DMCreateGlobalVector(da, &u_global));
  PetscCall(PetscObjectSetName((PetscObject)u_global, "Solution"));

  PetscScalar **u;
  PetscInt ibeg, jbeg, nlocx, nlocy;
  PetscCall(DMDAGetCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL));

  // Create an array with local coordinates.
  DM cda;
  Vec clocal;
  // array of local coordinates
  DMDACoor2d **alc;
  PetscCall(DMGetCoordinateDM(da, &cda));
  // This one has da, not cda
  PetscCall(DMGetCoordinatesLocal(da, &clocal));
  PetscCall(DMDAVecGetArrayRead(cda, clocal, &alc));

  PetscCall(DMDAVecGetArray(da, u_local, &u));

  // set the initial guess to zero, and boundary cells to g.
  // TODO Fill f[j][i] with the appropriate rhs value
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocy; ++i) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        // Then this point is on the boundary! Set it to g(x,y)
        u[j][i] = boundary_value(alc[j][i].x, alc[j][i].y);
      } else
        u[j][i] = 0.;
    }
  PetscCall(DMDAVecRestoreArray(da, u_local, &u));
  PetscCall(DMLocalToGlobal(da, u_local, INSERT_VALUES, u_global));
  // TODO save this initial condition please
  //  PetscCall(VecView(u_global, PETSC_VIEWER_STDOUT_WORLD));
  PetscPrintf(PETSC_COMM_WORLD, "Beginning iterations\n");

  // END OF IC SETUP. BEGIN ITER
  PetscReal maxdelta = 2. * eps;
  PetscInt iter = 0;
  PetscScalar **u_old;
  PetscScalar **u_new;
  while (iter < itermax && maxdelta > eps) {
    PetscCall(DMGlobalToLocalBegin(da, u_global, INSERT_VALUES, u_local));
    PetscCall(DMGlobalToLocalEnd(da, u_global, INSERT_VALUES, u_local));

    // Because u_local is a local array, ghost points will be accessible.
    PetscCall(DMDAVecGetArrayRead(da, u_local, &u_old));

    // Create an array for writing to u_global
    PetscCall(DMDAVecGetArrayWrite(da, u_global, &u_new));
    for (PetscInt j = jbeg; j < jbeg + nlocy; ++j) {
      for (PetscInt i = ibeg; i < ibeg + nlocx; ++i) {
        if (i * j == 0 || i == nx - 1 || j == ny - 1)
          // We don't want to update the values on the boundary layer
          // because Dirichlet BC.
          // TODO DMAddBoundary?
          continue;
        else {
          u_new[j][i] =
              0.25 * (h * h * rhs(alc[j][i].x, alc[j][i].y) + u_old[j - 1][i] + u_old[j + 1][i] +
                      u_old[j][i - 1] + u_old[j][i + 1]);
          maxdelta = std::max(abs(u_new[j][i] - u_old[j][i]), maxdelta);
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(da, u_local, &u_old));
    PetscCall(DMDAVecRestoreArrayWrite(da, u_global, &u_new));

    MPI_Allreduce(MPI_IN_PLACE, &maxdelta, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    ++iter;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iter, maxdelta = %d, %f\n", iter,
                          maxdelta));
    PetscCall(VecView(u_global, PETSC_VIEWER_STDOUT_WORLD));
  }
  // Always a good idea to destroy everything.
  PetscCall(VecDestroy(&u_global));
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

void JacobiSweep(PetscReal **u_old, PetscReal **u_new, PetscInt ibeg,
                 PetscInt jbeg, PetscInt nlocx, PetscInt nlocy) {}
