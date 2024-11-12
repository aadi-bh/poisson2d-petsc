# poisson2d-petsc
Uses PetSc to implement a parallel Jacobi solver for the Poisson equation in 2 dimensions.
Translation of Fortran version by Praveen C: https://github.com/cpraveen/parallel/mpi/poisson3d.f90

```
PETSC_DIR=<PETSC_DIR> PETSC_ARCH=<PETSC_ARCH> cmake .
cmake --build .
./poisson2d
```
