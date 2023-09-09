/* 
 * Solves 2D Poisson equation -∆u = f in Ω = [0,1]x[0,1]
 *                              u = g in ∂Ω
 *             using Jacobi iterationsh
 */
#include <petsc.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  char help[] = "Solves -∆u = f\n";
  // TODO PetscOptions this
  // Number of points along both directions.
  PetscInt N = 10;
  PetscInt iStart, iEnd;

  Vec U, b;
  Mat A;

  PetscInitialize(&argc, &argv, NULL, help);

  PetscOptionsBegin(PETSC_COMM_WORLD, "size_", "options for number of grid of points", "");
  PetscOptionsInt("-N", "size of grid along each axis", "main.cc", N, &N, NULL);
  PetscOptionsEnd();

  // Set b to be of length N^2
  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(b, PETSC_DECIDE, N*N);
  VecSetFromOptions(b);

  MatCreate(PETSC_COMM_WORLD, &A);                                                                                                                                    
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N*N, N*N);
  MatSetFromOptions(A);
  MatSetUp(A);
  MatGetOwnershipRange(A, &iStart, &iEnd);
  // Assemble the matrix
  // The discretisation is 
  // U[i][j+1] + U[i][j-1] + U[i-1][j] + U[i+1][j] - 4 U[i][j] = h^2 f_ij.
  // These are the coefficients that make the the coefficient matrix A.
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  //
  // U is a vector containing all U_ij, but serially, in lexicographic order.
  // So U_ij = N*i + j, where the grid is NxN.
  VecAssemblyBegin(b);
  //
  VecAssemblyEnd(b);
  // Then assemble the right-hand side
  VecDuplicate(b, &U);
  // then solve.
  // A while loop, calling sweep every time.

  // then output to VTK or something.
  //
  // Clean up.
  VecDestroy(&U);
  VecDestroy(&b);
  MatDestroy(&A);
  PetscFinalize();
  return 0;
}

