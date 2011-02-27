// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:
 * Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#ifndef _slepc_cxx_EPSolver_h
#define _slepc_cxx_EPSolver_h

#include <slepceps.h>

#include <core_library/Printable.h>
#include <core_library/UF.h>

#include <petsc_cxx/Matrix.h>
#include <petsc_cxx/Vector.h>

namespace slepc_cxx
{
    template < typename Atom >
    class EPSolver : public core_library::UF< const petsc_cxx::Matrix< Atom >&, void >, public core_library::Printable
    {
    public:
	EPSolver( EPSType type = EPSARNOLDI, MPI_Comm comm = PETSC_COMM_WORLD )
	{
	    EPSCreate( comm, &_solver );
	    EPSSetProblemType(_solver, EPS_HEP);
	    EPSSetFromOptions(_solver);
	    EPSSetType(_solver, type);
	}

	~EPSolver() { EPSDestroy( _solver ); }

	virtual void operator()( const petsc_cxx::Matrix< Atom >& A )
	{
	    MatGetVecs(A,PETSC_NULL,&_xr);
	    MatGetVecs(A,PETSC_NULL,&_xi);
	    EPSSetOperators(_solver, A, PETSC_NULL);
	    EPSSolve(_solver);
	}

	operator EPS() const { return _solver; }

	void printOn(std::ostream&) const
	{
	    const EPSType type;
	    PetscReal error, tol, re, im;
	    PetscScalar kr, ki;
	    PetscInt i, nev, maxit, its, nconv;

	    EPSGetIterationNumber(_solver,&its);
	    PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);
	    EPSGetType(_solver,&type);
	    PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
	    EPSGetDimensions(_solver,&nev,PETSC_NULL,PETSC_NULL);
	    PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);
	    EPSGetTolerances(_solver,&tol,&maxit);
	    PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);

	    EPSGetConverged(_solver,&nconv);
	    PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %d\n\n",nconv);

	    if (nconv>0)
		{
		    PetscPrintf(PETSC_COMM_WORLD,
				"           k          ||Ax-kx||/||kx||\n"
				"   ----------------- ------------------\n" );

		    for( i=0; i<nconv; i++ )
			{
			    EPSGetEigenpair(_solver,i,&kr,&ki,_xr,_xi);
			    EPSComputeRelativeError(_solver,i,&error);

#ifdef PETSC_USE_COMPLEX
			    re = PetscRealPart(kr);
			    im = PetscImaginaryPart(kr);
#else
			    re = kr;
			    im = ki;
#endif
			    if (im!=0.0)
				{
				    PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",re,im,error);
				}
			    else
				{
				    PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n",re,error);
				}
			}
		    PetscPrintf(PETSC_COMM_WORLD,"\n" );
		}
	}

    private:
	EPS _solver;
	Vec _xr;
	Vec _xi;
    };
}

#endif // !_slepc_cxx_EPSolver_h
