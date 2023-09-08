#include <petsc.h>

int main(int argc, char* argv[])
{
  char help[] = "Solves -∆u = f\n";
  PetscInitialize(&argc, &argv, NULL, help);
  PetscPrintf(PETSC_COMM_WORLD, "Hello PetSc world!\n");
  PetscFinalize();
  return 0;
}

