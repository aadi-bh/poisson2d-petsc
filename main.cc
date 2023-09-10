/*
 * Solves 2D Poisson equation -∆u = f = rhs            in Ω = [0,1]x[0,1]
 *                              u = g = boundary_value in ∂Ω
 *             using Jacobi iterations.
 */
#include <cstddef>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <fstream>
#include <petsc.h>

PetscErrorCode write_rectilinear_grid(DM da, Vec u_global, DM cda, Vec clocal,
                            PetscInt iter, PetscReal t, PetscInt c);

PetscReal rhs(PetscInt x, PetscInt y) { return 4. * (pow(x, 4) + pow(y, 4)); }

PetscReal boundary_value(PetscReal x, PetscReal y) { return x * x + y * y; }

int main(int argc, char *argv[]) {
  char help[] = "Solves -∆u = f\n";
  // Number of points along both directions.
  PetscInt N = 10, nx, ny;
  // TODO Make an option for these too.
  const PetscReal eps = 1e-5;
  const PetscInt itermax = 10000;
  PetscReal h;
  constexpr PetscReal xmin = -1.;
  constexpr PetscReal xmax = 1.;
  constexpr PetscReal ymin = -1.;
  constexpr PetscReal ymax = 1;
  // TODO Error-checking
  // DM stands for "distributed mesh"
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
  //  PetscReal dy = (ymax - ymin) / (PetscReal)(ny);
  h = dx;

  PetscCall(DMCreateLocalVector(da, &u_local));
  PetscCall(DMCreateGlobalVector(da, &u_global));
  PetscCall(PetscObjectSetName((PetscObject)u_global, "Global solution"));
  PetscCall(PetscObjectSetName((PetscObject)u_local, "Local solution"));

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
  // TODO Fill f[j][i] with the appropriate rhs value. Why compute the same
  // thing over and over again?
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocy; ++i) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        // Then this point is on the boundary! Set it to g(x,y)
        u[j][i] = boundary_value(alc[j][i].x, alc[j][i].y);
      } else
        u[j][i] = 0.;
    }
  write_rectilinear_grid(da, u_global, cda, clocal, 0, 0, 0);
  PetscCall(DMDAVecRestoreArray(da, u_local, &u));
  PetscCall(DMLocalToGlobal(da, u_local, INSERT_VALUES, u_global));
  // TODO save this initial condition instead of printing it out.

  // END OF IC SETUP. BEGIN ITER
  PetscReal maxdelta = 2. * eps;
  PetscInt iter = 0;
  PetscScalar **u_old;
  PetscScalar **u_new;
  while (iter < itermax && maxdelta > eps) {
    // Transfer data from global to local array.
    PetscCall(DMGlobalToLocalBegin(da, u_global, INSERT_VALUES, u_local));
    PetscCall(DMGlobalToLocalEnd(da, u_global, INSERT_VALUES, u_local));

    // Make u_old point to the data in the u_local array
    PetscCall(DMDAVecGetArrayRead(da, u_local, &u_old));

    maxdelta = 0;
    // Get a pointer to the portion of memory to put the new data into.
    PetscCall(DMDAVecGetArrayWrite(da, u_global, &u_new));
    for (PetscInt j = jbeg; j < jbeg + nlocy; ++j) {
      for (PetscInt i = ibeg; i < ibeg + nlocx; ++i) {
        if (i * j == 0 || i == nx - 1 || j == ny - 1)
          // We don't want to update the values on the boundary layer
          // because Dirichlet BC.
          // TODO DMAddBoundary?
          continue;
        else {
          // Because u_local is a local array, ghost points will be accessible.
          u_new[j][i] =
              0.25 * (h * h * rhs(alc[j][i].x, alc[j][i].y) + u_old[j - 1][i] +
                      u_old[j + 1][i] + u_old[j][i - 1] + u_old[j][i + 1]);
          maxdelta = std::max(abs(u_new[j][i] - u_old[j][i]), maxdelta);
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(da, u_local, &u_old));
    PetscCall(DMDAVecRestoreArrayWrite(da, u_global, &u_new));

    std::cout << maxdelta << std::endl;
    MPI_Allreduce(MPI_IN_PLACE, &maxdelta, 1, MPI_DOUBLE_PRECISION, MPI_MAX,
                PETSC_COMM_WORLD);
    ++iter;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iter, maxdelta = %6d, %f\n", iter,
                          maxdelta));
  }
  VecView(u_global, PETSC_VIEWER_STDOUT_WORLD);
  write_rectilinear_grid(da, u_global, cda, clocal, iter, 0, 0);
  PetscCall(DMDAVecRestoreArrayRead(cda, clocal, &alc));
  // Always a good idea to destroy everything.
  PetscCall(VecDestroy(&u_global));
  // Do NOT destroy clocal, it is a borrowed vector. It goes when the DM goes.
  // Also do NOT destroy &cda. It's causing a segfault in this line, or in the
  // da line below.
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

// TODO Separate everything into functions. Sweep at least.
// from
// https://github.com/aadi-bh/parallel/blob/60bfbfa85302bd9cf97ccc123c99032cfb496173/mpi/poisson3d.cc#L493
/*
 * Writes out a VTK file with the given slice of an array
 */
PetscErrorCode write_rectilinear_grid(DM da, Vec u_global, DM cda, Vec clocal,
                            PetscInt iter, PetscReal t, PetscInt c) {
  using namespace std;
  Vec u_local;
  PetscScalar **sol;
  PetscCall(DMGetLocalVector(da, &u_local));
  PetscCall(DMGlobalToLocal(da, u_global, INSERT_VALUES, u_local));
  PetscCall(DMDAVecGetArrayRead(da, u_local, &sol));

  DMDACoor2d **alc;
  PetscCall(DMDAVecGetArrayRead(cda, clocal, &alc));

  PetscInt ibeg, jbeg, nlocx, nlocy;
  // TODO Get ghost coordinates so that we can plot those too
  PetscCall(DMDAGetCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL));
  PetscInt index_range[2][2];
  index_range[0][0] = ibeg;
  index_range[0][1] = ibeg + nlocx;
  index_range[1][0] = jbeg;
  index_range[1][1] = jbeg + nlocy;

  int n[2];
  for (int dir = 0; dir < 2; ++dir)
    n[dir] = index_range[dir][1] - index_range[dir][0];

  PetscInt id;
  MPI_Comm_rank(PETSC_COMM_WORLD, &id);
  ofstream fout;
  char filename[64];
  snprintf(filename, 64, "sol-%d-%d.vtk", id, iter);
  fout.open(filename);

  fout << "# vtk DataFile Version 3.0" << endl;
  fout << "Cartesian grid" << endl;
  fout << "ASCII" << endl;
  fout << "DATASET RECTILINEAR_GRID" << endl;
  fout << "FIELD FieldData 2" << endl;
  fout << "TIME 1 1 double" << endl;
  fout << t << endl;
  fout << "CYCLE 1 1 int" << endl;
  fout << c << endl;
  fout << "DIMENSIONS " << n[0] << " " << n[1] << " " << 1 << endl;
  fout << "X_COORDINATES " << n[0] << " float" << endl;
  
  for (int i = ibeg; i < ibeg + nlocx; ++i)
    fout << alc[0][i].x << " ";
  fout << endl;

  fout << "Y_COORDINATES " << n[1] << " float" << endl;
  for (int j = jbeg; j < jbeg + nlocy; ++j)
    fout << alc[j][0].y << " ";
  fout << endl;
   fout << "Z_COORDINATES " << 1 << " float" << endl;
   fout << 0.0 << endl;
  fout << "POINT_DATA " << n[0] * n[1] << endl;
  fout << "SCALARS density double" << endl;
  fout << "LOOKUP_TABLE default" << endl;
  for (int j = index_range[1][0]; j < index_range[1][1]; ++j) {
    for (int i = index_range[0][0]; i < index_range[0][1]; ++i)
      fout << sol[j][i] << " ";
    fout << endl;
  }
  fout << endl;
  fout.close();
  cout << filename << endl;
//  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s\n", filename));
  PetscCall(DMDAVecRestoreArrayRead(da, u_local, &sol));
  PetscCall(DMRestoreLocalVector(da, &u_local));
  return PetscErrorCode(PETSC_SUCCESS);
}
