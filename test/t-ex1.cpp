// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Authors: Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

// Binding of the example available on file:///usr/share/doc/petsc3.1-doc/src/vec/vec/examples/tutorials/ex1.c.html

#include <slepc_cxx/slepc_cxx>

static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

typedef petsc_cxx::Scalar T;

int main(int ac, char** av)
{
    slepc_cxx::Parser parser(ac, av, help);
    petsc_cxx::Context context( parser );

    slepc_cxx::EPSolver<T> eps;

    PetscInt n=30, i, Istart, Iend, col[3];
    PetscTruth FirstBlock=PETSC_FALSE, LastBlock=PETSC_FALSE;
    PetscScalar value[3];

    petsc_cxx::Matrix<T> A(n);

    MatGetOwnershipRange(A,&Istart,&Iend);

    if (Istart==0) FirstBlock=PETSC_TRUE;
    if (Iend==n) LastBlock=PETSC_TRUE;

    value[0]=-1.0; value[1]=2.0; value[2]=-1.0;

    for( i = (FirstBlock? Istart+1: Istart); i < (LastBlock? Iend-1: Iend); i++ )
	{
	    col[0]=i-1; col[1]=i; col[2]=i+1;
	    MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
	}
    if (LastBlock)
	{
	    i=n-1; col[0]=n-2; col[1]=n-1;
	    MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
	}
    if (FirstBlock)
	{
	    i=0; col[0]=0; col[1]=1; value[0]=2.0; value[1]=-1.0;
	    MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
	}

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    eps(A);

    std::cout << eps;

    return 0;
}
