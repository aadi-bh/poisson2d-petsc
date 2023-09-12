/*
 * Solves 2D Poisson equation -∆u = f = rhs            in Ω = [0,1]x[0,1]
 *                              u = g = boundary_value in ∂Ω
 *             using Jacobi iterations.
 *
 *  Date: Friday, September 8th, 2023.
 */
#include <fstream>
#include <petsc.h>
#include <stdio.h>

PetscErrorCode viewerOutput(DM da, Vec u, PetscInt iter);
PetscErrorCode write_rectilinear_grid(DM da, Vec u_global, DM cda, Vec clocal,
                                      PetscInt iter, PetscReal t, PetscInt c);

PetscReal rhs(PetscInt x, PetscInt y) { return 4. * (pow(x, 4) + pow(y, 4)); }
PetscReal boundary_value(PetscReal x, PetscReal y) { return x * x + y * y; }

int main(int argc, char *argv[])
{
  char                help[]  = "Solves -∆u = f\n";
  // Number of points along both directions.
  PetscInt            N       = 100;
  // TODO Make an option for these too.
  PetscReal     eps     = 1e-5;
  PetscInt      itermax = 10000;
  PetscReal xmin = -1.;
  PetscReal xmax = 1.;
  PetscReal ymin = -1.;
  PetscReal ymax = 1;
  // TODO Error-checking
  // DM stands for "distributed mesh"
  // da stands for "distributed array"
  DM                  da;
  Vec                 u_local, u_global;
  PetscInt nx, ny;
  PetscReal           dx, dy, h;

  // This calls MPI_Init unless we have already.
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // PetscOptionsBegin and ...End are macros. They don't have return
  // values, so they can't be used with PetscCall.
  PetscOptionsBegin(PETSC_COMM_WORLD, "jacobi_",
                    "options for number of grid of points", "");
  // TODO Put defaults here, and declare them as uninit'd consts above.
  PetscCall(PetscOptionsInt("-N", "size of grid along each axis", "main.cc", N,
                            &N, NULL));
  PetscCall(PetscOptionsInt("-itermax", "Maximum number of iterations", "main.cc", itermax,
                            &itermax, NULL));
  PetscCall(PetscOptionsReal("-eps", "Step size to halt at", "main.cc", eps,
                            &eps, NULL));
  PetscOptionsEnd();
  // Create a 2D Distributed Mesh of DA type.
  // We choose BOUNDARY_NONE because we don't want a layer of ghosts
  // around the whole domain. We know the values on the boundary.
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                         /* boundary types */ DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE,
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

  // Sets uniform coordinates for the grid. Z-values are ignored in 2D
  PetscCall(DMDASetUniformCoordinates(da, xmin, xmax, ymin, ymax, 0., 0.));

  // Find h
  PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL));

  // TODO is this necessary?
  if (nx != ny)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "nx not equal to ny\n"));

  dx = (xmax - xmin) / (PetscReal)(nx);
  dy = (ymax - ymin) / (PetscReal)(ny);
  h            = dx;

  PetscCall(DMCreateLocalVector(da, &u_local));
  PetscCall(DMCreateGlobalVector(da, &u_global));
  PetscCall(PetscObjectSetName((PetscObject)u_global, "Poisson solution"));
  // TODO The name should have the rank id in it. 
  PetscCall(PetscObjectSetName((PetscObject)u_local, "Local solution"));

  PetscInt      ibeg, jbeg, nlocx, nlocy;
  PetscCall(DMDAGetCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL));

  // Create an array with local coordinates.
  DM           cda;
  Vec          clocal;
  // DMDACoor2d is just a struct of 2 PetscScalars x and y.
  DMDACoor2d **alc;
  PetscCall(DMGetCoordinateDM(da, &cda));
  // This one has da, not cda
  PetscCall(DMGetCoordinatesLocal(da, &clocal));
  PetscCall(DMDAVecGetArrayRead(cda, clocal, &alc));

  PetscScalar **u;
  // VecGetArray will point u to the data stored inside u_local.
  PetscCall(DMDAVecGetArray(da, u_local, &u));
  // set the initial guess to zero, and boundary cells to boundary_value.
  // TODO Store the rhs value at each [j][i]. Why compute again on every iteration?
  for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    for (PetscInt i = ibeg; i < ibeg + nlocx; ++i)
    {
      // TODO How do I check whether this point is on the boundary 
      // when the domain gets more complex?
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
      {
        // Then this point is on the boundary! Set it to g(x,y)
        u[j][i] = boundary_value(alc[j][i].x, alc[j][i].y);
      }
      else
        u[j][i] = 0.;
    }
  // Zeroes out the pointer u, among other things.
  PetscCall(DMDAVecRestoreArray(da, u_local, &u));
  // Updates the global vector with the local values.
  PetscCall(DMLocalToGlobal(da, u_local, INSERT_VALUES, u_global));
  // Save initial condition.
  write_rectilinear_grid(da, u_global, cda, clocal, 0, 0, 0);

  /* ----------------------------
   * END OF SETUP
   * BEGIN ITERATIONS
   * ----------------------------
   */
  PetscReal     maxdelta = 2. * eps;
  PetscInt      iter     = 0;
  PetscScalar **u_old;
  PetscScalar **u_new;
  while (iter < itermax && maxdelta > eps)
  {
    // Transfer data from global to local array.
    PetscCall(DMGlobalToLocalBegin(da, u_global, INSERT_VALUES, u_local));
    PetscCall(DMGlobalToLocalEnd(da, u_global, INSERT_VALUES, u_local));

    // Make u_old point to the data in the u_local array
    PetscCall(DMDAVecGetArrayRead(da, u_local, &u_old));

    // Get a pointer to the portion of memory where we will be putting the new solution.
    PetscCall(DMDAVecGetArrayWrite(da, u_global, &u_new));
    maxdelta = 0;
    for (PetscInt j = jbeg; j < jbeg + nlocy; ++j)
    {
      for (PetscInt i = ibeg; i < ibeg + nlocx; ++i)
      {
        if (i * j == 0 || i == nx - 1 || j == ny - 1)
          // We don't want to update the values on the boundary layer
          // because Dirichlet BC.
          // TODO DMAddBoundary?
          continue;
        else
        {
          // Because u_local is a local array, ghost points will be accessible.
          u_new[j][i] =
              0.25
              * (h * h * rhs(alc[j][i].x, alc[j][i].y) + u_old[j - 1][i]
                 + u_old[j + 1][i] + u_old[j][i - 1] + u_old[j][i + 1]);
          maxdelta = std::max(abs(u_new[j][i] - u_old[j][i]), maxdelta);
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(da, u_local, &u_old));
    PetscCall(DMDAVecRestoreArrayWrite(da, u_global, &u_new));

    ++iter;
    // Get the global maxdelta.
    MPI_Allreduce(MPI_IN_PLACE, &maxdelta, 1, MPI_DOUBLE_PRECISION, MPI_MAX,
                  PETSC_COMM_WORLD);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iter, maxdelta = %6d, %f\n", iter,
                          maxdelta));
  }
  // viewerOutput(da, u_global, iter);
  // Save the last one.
  write_rectilinear_grid(da, u_global, cda, clocal, iter, 0, 0);

  /* ----------------
   * END OF ITER
   * BEGIN CLEANUP
   * ----------------
   */
  PetscCall(DMDAVecRestoreArrayRead(cda, clocal, &alc));
  // Always a good idea to destroy everything.
  PetscCall(VecDestroy(&u_global));
  // Do NOT destroy clocal, it is a borrowed vector. It goes when the DM goes.
  // Also do NOT destroy cda. It's causing a segfault somewhere below.
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*
 * Uses the VecView function to create a file in the required format
 */
PetscErrorCode viewerOutput(DM da, Vec u_global, PetscInt iter)
{
#if defined(PETSC_HAVE_HDF5)
  PetscFunctionBeginUser;
  PetscInt id;
  MPI_Comm_rank(PETSC_COMM_WORLD, &id);
  char filename[64];
  snprintf(filename, 64, "sol-%d-%d.h5", id, iter);

  Vec u_local;
  PetscCall(DMGetLocalVector(da, &u_local));
  PetscCall(DMGlobalToLocal(da, u_global, INSERT_VALUES, u_local));

  PetscViewer viewer;
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERHDF5));
  PetscCall(VecView(u_local, viewer));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Saved to %s.\n", filename));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  PetscFunctionReturn(PETSC_FAILURE);
#endif
}

// from
// https://github.com/aadi-bh/parallel/blob/60bfbfa85302bd9cf97ccc123c99032cfb496173/mpi/poisson3d.cc#L493
/*
 * Writes out a VTK file with the local data in u_global. Corners and coordinates provided by clocal.
 */
PetscErrorCode write_rectilinear_grid(DM da, Vec u_global, DM cda, Vec clocal,
                                      PetscInt iter, PetscReal t, PetscInt c)
{
  PetscInt id;
  MPI_Comm_rank(PETSC_COMM_WORLD, &id);
  char filename[64];
  snprintf(filename, 64, "sol-%d-%d.vtk", id, iter);


  Vec           u_local;
  PetscScalar **sol;
  PetscCall(DMGetLocalVector(da, &u_local));
  // This fills u_local with the latest values, including ghosts.
  PetscCall(DMGlobalToLocal(da, u_global, INSERT_VALUES, u_local));
  PetscCall(DMDAVecGetArrayRead(da, u_local, &sol));

  DMDACoor2d **alc;
  PetscCall(DMDAVecGetArrayRead(cda, clocal, &alc));

  PetscInt ibeg, jbeg, nlocx, nlocy;
  // We are going to save the latest ghost values too.
  PetscCall(DMDAGetGhostCorners(da, &ibeg, &jbeg, NULL, &nlocx, &nlocy, NULL));

  // Set the data name to the name given to the vector.
  const char *buffer;
  PetscObjectGetName((PetscObject)u_global, &buffer);
  char objName[64];
  strncpy(objName, buffer, 64);
  for (int c = 0, n = strlen(objName); c < n; ++c)
  {
    if (objName[c] == ' ')
      objName[c] = '_';
  }

  using namespace std;
  ofstream fout;
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
  // 1 for Z because this is 2D
  fout << "DIMENSIONS " << nlocx << " " << nlocy << " " << 1 << endl;

  fout << "X_COORDINATES " << nlocx << " float" << endl;
  for (int i = ibeg; i < ibeg + nlocx; ++i)
    fout << alc[jbeg][i].x << " ";
  fout << endl;

  fout << "Y_COORDINATES " << nlocy << " float" << endl;
  for (int j = jbeg; j < jbeg + nlocy; ++j)
    fout << alc[j][ibeg].y << " ";
  fout << endl;

  fout << "Z_COORDINATES " << 1 << " float" << endl;
  fout << 0.0 << endl;

  fout << "POINT_DATA " << nlocx * nlocy << endl;
  fout << "SCALARS " << objName << " double" << endl;
  fout << "LOOKUP_TABLE default" << endl;
  for (int j = jbeg; j < nlocy; ++j)
  {
    for (int i = ibeg; i < nlocx; ++i)
      fout << sol[j][i] << " ";
    fout << endl;
  }
  fout << endl;
  fout.close();

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s\n", filename));
  PetscCall(DMDAVecRestoreArrayRead(da, u_local, &sol));
  PetscCall(DMRestoreLocalVector(da, &u_local));
  PetscCall(DMDAVecRestoreArrayRead(cda, clocal, &alc));

  return PetscErrorCode(PETSC_SUCCESS);
}
