/*

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Algorithm:

       Arnoldi method with explicit restart and deflation.

   References:

       [1] "Arnoldi Methods in SLEPc", SLEPc Technical Report STR-4,
           available at http://www.grycap.upv.es/slepc.

   Last update: Feb 2009

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  PetscTruth delayed;
} EPS_ARNOLDI;

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_ARNOLDI"
PetscErrorCode EPSSetUp_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev");
  }
  else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(N,eps->nev+eps->mpd);
  }
  else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
    else { eps->mpd = 500; eps->ncv = PetscMin(N,eps->nev+eps->mpd); }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(1,"The value of ncv must not be larger than nev+mpd");
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY))
    SETERRQ(1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
    PetscInfo(eps,"Warning: parameter mpd ignored\n");
  }
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSBasicArnoldi"
/*
   EPSBasicArnoldi - Computes an m-step Arnoldi factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * H = f * e_m^T

   where the columns of V are the Arnoldi vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, f is the residual vector and e_m is
   the m-th vector of the canonical basis. The vector f is B-orthogonal to
   the columns of V. On exit, beta contains the B-norm of f and the next
   Arnoldi vector can be computed as v_{m+1} = f / beta.
*/
PetscErrorCode EPSBasicArnoldi(EPS eps,PetscTruth trans,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m = *M;
  PetscReal      norm;
  PetscScalar    *swork = PETSC_NULL,*hwork = PETSC_NULL;

  PetscFunctionBegin;
  if (eps->nds+m > 100) { ierr = PetscMalloc((eps->nds+m)*sizeof(PetscScalar),&swork);CHKERRQ(ierr); }
  if (eps->nds > 0) { ierr = PetscMalloc((eps->nds+m)*sizeof(PetscScalar),&hwork);CHKERRQ(ierr); }

  for (j=k;j<m-1;j++) {
    if (trans) { ierr = STApplyTranspose(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    else { ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    if (eps->nds > 0) {
      ierr = IPOrthogonalize(eps->ip,eps->nds+j+1,PETSC_NULL,eps->DSV,V[j+1],hwork,&norm,breakdown,eps->work[0],swork);CHKERRQ(ierr);
      for (i=0;i<=j;i++) 
        H[ldh*j+i] = hwork[eps->nds+i];
    } else {
      ierr = IPOrthogonalize(eps->ip,j+1,PETSC_NULL,V,V[j+1],H+ldh*j,&norm,breakdown,eps->work[0],swork);CHKERRQ(ierr);
    }
    H[j+1+ldh*j] = norm;
    if (*breakdown) {
      *M = j+1;
      *beta = norm;
      if (swork) { ierr = PetscFree(swork);CHKERRQ(ierr); }
      if (hwork) { ierr = PetscFree(hwork);CHKERRQ(ierr); }
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  if (trans) { ierr = STApplyTranspose(eps->OP,V[m-1],f);CHKERRQ(ierr); }
  else { ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr); }
  ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0],swork);CHKERRQ(ierr);
  ierr = IPOrthogonalize(eps->ip,m,PETSC_NULL,V,f,H+ldh*(m-1),beta,PETSC_NULL,eps->work[0],swork);CHKERRQ(ierr);
  
  if (swork) { ierr = PetscFree(swork);CHKERRQ(ierr); }
  if (hwork) { ierr = PetscFree(hwork);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDelayedArnoldi"
/*
   EPSDelayedArnoldi - This function is equivalent to EPSBasicArnoldi but
   performs the computation in a different way. The main idea is that
   reorthogonalization is delayed to the next Arnoldi step. This version is
   more scalable but in some cases convergence may stagnate.
*/
PetscErrorCode EPSDelayedArnoldi(EPS eps,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  Vec            w,u,t;
  PetscScalar    shh[100],*lhh,dot,dot2;
  PetscReal      norm1=0.0,norm2;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0],PETSC_NULL);CHKERRQ(ierr);

    ierr = IPMInnerProductBegin(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = IPMInnerProductBegin(eps->ip,V[j],j,V,lhh);CHKERRQ(ierr);
      ierr = IPInnerProductBegin(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = IPNormBegin(eps->ip,u,&norm2);CHKERRQ(ierr); 
      ierr = VecDotBegin(u,V[j-2],&dot2);CHKERRQ(ierr);
    }
    
    ierr = IPMInnerProductEnd(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = IPMInnerProductEnd(eps->ip,V[j],j,V,lhh);CHKERRQ(ierr);
      ierr = IPInnerProductEnd(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = IPNormEnd(eps->ip,u,&norm2);CHKERRQ(ierr);
      ierr = VecDotEnd(u,V[j-2],&dot2);CHKERRQ(ierr);
      if (PetscAbsScalar(dot2/norm2) > PETSC_MACHINE_EPSILON) {
        *breakdown = PETSC_TRUE;
      *M = j-1;
      *beta = norm2;

      if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
      ierr = VecDestroy(w);CHKERRQ(ierr);
      ierr = VecDestroy(u);CHKERRQ(ierr);
      ierr = VecDestroy(t);CHKERRQ(ierr);
      PetscFunctionReturn(0);
      }
    }
    
    if (j>k) {      
      norm1 = sqrt(PetscRealPart(dot));
      for (i=0;i<j;i++)
      H[ldh*j+i] = H[ldh*j+i]/norm1;
      H[ldh*j+j] = H[ldh*j+j]/dot;
      
      ierr = VecCopy(V[j],t);CHKERRQ(ierr);
      ierr = VecScale(V[j],1.0/norm1);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm1);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+ldh*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j>k) {
      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
      ierr = VecAXPY(t,-1.0,w);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[ldh*(j-1)+i] += lhh[i];
    }

    if (j>k+1) {
      ierr = VecCopy(u,V[j-1]);CHKERRQ(ierr);
      ierr = VecScale(V[j-1],1.0/norm2);CHKERRQ(ierr);
      H[ldh*(j-2)+j-1] = norm2;
    }

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
      ierr = VecCopy(t,u);CHKERRQ(ierr);
    }
  }

  ierr = IPNorm(eps->ip,t,&norm2);CHKERRQ(ierr);
  ierr = VecScale(t,1.0/norm2);CHKERRQ(ierr);
  ierr = VecCopy(t,V[m-1]);CHKERRQ(ierr);
  H[ldh*(m-2)+m-1] = norm2;

  ierr = IPMInnerProduct(eps->ip,f,m,V,lhh);CHKERRQ(ierr);
  
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,m,lhh,V);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[ldh*(m-1)+i] += lhh[i];

  ierr = IPNorm(eps->ip,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;
  
  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(t);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDelayedArnoldi1"
/*
   EPSDelayedArnoldi1 - This function is similar to EPSDelayedArnoldi1,
   but without reorthogonalization (only delayed normalization).
*/
PetscErrorCode EPSDelayedArnoldi1(EPS eps,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  Vec            w;
  PetscScalar    dot;
  PetscReal      norm=0.0;

  PetscFunctionBegin;
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL,eps->work[0],PETSC_NULL);CHKERRQ(ierr);

    ierr = IPMInnerProductBegin(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = IPInnerProductBegin(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    
    ierr = IPMInnerProductEnd(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = IPInnerProductEnd(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    
    if (j>k) {      
      norm = sqrt(PetscRealPart(dot));
      ierr = VecScale(V[j],1.0/norm);CHKERRQ(ierr);
      H[ldh*(j-1)+j] = norm;

      for (i=0;i<j;i++)
      H[ldh*j+i] = H[ldh*j+i]/norm;
      H[ldh*j+j] = H[ldh*j+j]/dot;      
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+ldh*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  ierr = IPNorm(eps->ip,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;
  
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ArnoldiResiduals"
/*
   EPSArnoldiResiduals - Computes the 2-norm of the residual vectors from
   the information provided by the m-step Arnoldi factorization,

                    OP * V - V * H = f * e_m^T

   For the approximate eigenpair (k_i,V*y_i), the residual norm is computed as
   |beta*y(end,i)| where beta is the norm of f and y is the corresponding 
   eigenvector of H.
*/
PetscErrorCode ArnoldiResiduals(PetscScalar *H,PetscInt ldh_,PetscScalar *U,PetscReal beta,PetscInt nconv,PetscInt ncv_,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscScalar *work)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   mout,info,ldh,ncv;
  PetscScalar    *Y=work+4*ncv_;
  PetscReal      w;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork=(PetscReal*)(work+3*ncv_);
#endif

  PetscFunctionBegin;
  ldh = PetscBLASIntCast(ldh_);
  ncv = PetscBLASIntCast(ncv_);

  /* Compute eigenvectors Y of H */
  ierr = PetscMemcpy(Y,U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","B",PETSC_NULL,&ncv,H,&ldh,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,&info);
#else
  LAPACKtrevc_("R","B",PETSC_NULL,&ncv,H,&ldh,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&info);
#endif
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  /* Compute residual norm estimates as beta*abs(Y(m,:)) */
  for (i=nconv;i<ncv;i++) { 
#if !defined(PETSC_USE_COMPLEX)
    if (eigi[i] != 0 && i<ncv-1) {
      errest[i] = beta*SlepcAbsEigenvalue(Y[i*ncv+ncv-1],Y[(i+1)*ncv+ncv-1]);
      w = SlepcAbsEigenvalue(eigr[i],eigi[i]);
      if (w > errest[i]) 
      errest[i] = errest[i] / w;
      errest[i+1] = errest[i];
      i++;
    } else
#endif
    {
      errest[i] = beta*PetscAbsScalar(Y[i*ncv+ncv-1]);
      w = PetscAbsScalar(eigr[i]);
      if (w > errest[i]) 
      errest[i] = errest[i] / w;
    }
  }  
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedArnoldi"
/*
   EPSProjectedArnoldi - Solves the projected eigenproblem.

   On input:
     S is the projected matrix (leading dimension is lds)

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the corresponding Schur vectors (order n, leading dimension n)
*/
PetscErrorCode EPSProjectedArnoldi(EPS eps,PetscScalar *S,PetscInt lds,PetscScalar *Q,PetscInt n)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  /* Initialize orthogonal matrix */
  ierr = PetscMemzero(Q,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) 
    Q[i*(n+1)] = 1.0;
  /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
  ierr = EPSDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);
  /* Sort the remaining columns of the Schur form */
  if (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = EPSSortDenseSchurTarget(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->target,eps->which);CHKERRQ(ierr); 
  } else {
    ierr = EPSSortDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi,eps->which);CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSUpdateVectors"
/*
   EPSUpdateVectors - Computes approximate Schur vectors (or eigenvectors) by
   either Ritz extraction (U=U*Q) or refined Ritz extraction 

   On input:
     n is the size of U
     U is the orthogonal basis of the subspace used for projecting
     s is the index of the first vector computed
     e+1 is the index of the last vector computed
     Q contains the corresponding Schur vectors of the projected matrix (size n x n, leading dimension ldq)
     H is the (extended) projected matrix (size n+1 x n, leading dimension ldh)

   On output:
     v is the resulting vector
*/
PetscErrorCode EPSUpdateVectors(EPS eps,PetscInt n_,Vec *U,PetscInt s,PetscInt e,PetscScalar *Q,PetscInt ldq,PetscScalar *H,PetscInt ldh_)
{
#if defined(PETSC_MISSING_LAPACK_GESVD) 
  SETERRQ(PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscTruth     isrefined;
  PetscInt       i,j,k;
  PetscBLASInt   n1,lwork,idummy=1,info,n=n_,ldh=ldh_;
  PetscScalar    *B,sdummy,*work;
  PetscReal      *sigma;

  PetscFunctionBegin;
  isrefined = (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  if (isrefined) {
    /* Refined Ritz extraction */
    n1 = n+1;
    ierr = PetscMalloc(n1*n*sizeof(PetscScalar),&B);CHKERRQ(ierr);
    ierr = PetscMalloc(6*n*sizeof(PetscReal),&sigma);CHKERRQ(ierr);
    lwork = 10*n;
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    
    for (k=s;k<e;k++) {
      /* copy H to B */
      for (i=0;i<=n;i++) {
        for (j=0;j<n;j++) {
          B[i+j*n1] = H[i+j*ldh];
        }
      }
      /* subtract ritz value from diagonal of B^ */
      for (i=0;i<n;i++) {
        B[i+i*n1] -= eps->eigr[k];  /* MISSING: complex case */
      }
      /* compute SVD of [H-mu*I] */
  #if !defined(PETSC_USE_COMPLEX)
      LAPACKgesvd_("N","O",&n1,&n,B,&n1,sigma,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&info);
  #else
      LAPACKgesvd_("N","O",&n1,&n,B,&n1,sigma,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,sigma+n,&info);
  #endif
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
      /* the smallest singular value is the new error estimate */
      eps->errest[k] = sigma[n-1];
      /* update vector with right singular vector associated to smallest singular value */
      for (i=0;i<n;i++)
        Q[k*ldq+i] = B[n-1+i*n1];
    }
    /* free workspace */
    ierr = PetscFree(B);CHKERRQ(ierr);
    ierr = PetscFree(sigma);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  /* Ritz extraction: v = U*q */
  ierr = SlepcUpdateVectors(n_,U,s,e,Q,ldq,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
PetscErrorCode EPSSolve_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,nv;
  Vec            f=eps->work[1];
  PetscScalar    *H=eps->T,*U,*g,*work,*Hcopy;
  PetscReal      beta,gnorm;
  PetscTruth     breakdown;
  IPOrthogonalizationRefinementType orthog_ref;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscMemzero(eps->T,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
  ierr = PetscMalloc((eps->ncv+4)*eps->ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  if (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&g);CHKERRQ(ierr);
  }
  if (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = PetscMalloc((eps->ncv+1)*eps->ncv*sizeof(PetscScalar),&Hcopy);CHKERRQ(ierr);
  }
  
  ierr = IPGetOrthogonalization(eps->ip,PETSC_NULL,&orthog_ref,PETSC_NULL);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    if (!arnoldi->delayed) {
      ierr = EPSBasicArnoldi(eps,PETSC_FALSE,H,eps->ncv,eps->V,eps->nconv,&nv,f,&beta,&breakdown);CHKERRQ(ierr);
    } else if (orthog_ref == IP_ORTH_REFINE_NEVER) {
      ierr = EPSDelayedArnoldi1(eps,H,eps->ncv,eps->V,eps->nconv,&nv,f,&beta,&breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSDelayedArnoldi(eps,H,eps->ncv,eps->V,eps->nconv,&nv,f,&beta,&breakdown);CHKERRQ(ierr);
    }

    if (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC) {
      ierr = PetscMemcpy(Hcopy,H,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
      for (i=0;i<nv-1;i++) Hcopy[nv+i*eps->ncv] = 0.0; 
      Hcopy[nv+(nv-1)*eps->ncv] = beta;
    }

    /* Compute translation of Krylov decomposition if harmonic extraction used */ 
    if (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC) {
      ierr = EPSTranslateHarmonic(nv,H,eps->ncv,eps->target,(PetscScalar)beta,g,work);CHKERRQ(ierr);
    }

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedArnoldi(eps,H,eps->ncv,U,nv);CHKERRQ(ierr);
    ierr = ArnoldiResiduals(H,eps->ncv,U,beta,eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);
    
    /* Fix residual norms if harmonic */
    if (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC) {
      gnorm = 0.0;
      for (i=0;i<nv;i++)
        gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));
      for (i=eps->nconv;i<nv;i++)
        eps->errest[i] *= sqrt(1.0+gnorm);
    }

    /* Lock converged eigenpairs and update the corresponding vectors,
       including the restart vector: V(:,idx) = V*U(:,idx) */
    k = eps->nconv;
    while (k<nv && eps->errest[k]<eps->tol) k++;
    ierr = EPSUpdateVectors(eps,nv,eps->V,eps->nconv,PetscMin(k+1,nv),U,nv,Hcopy,eps->ncv);CHKERRQ(ierr);
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);
    if (breakdown) {
      PetscInfo2(eps,"Breakdown in Arnoldi method (it=%i norm=%g)\n",eps->its,beta);
      ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
      PetscInfo(eps,"Unable to generate more start vectors\n");
      }
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = PetscFree(g);CHKERRQ(ierr);
  }
  if (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = PetscFree(Hcopy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_ARNOLDI"
PetscErrorCode EPSSetFromOptions_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ARNOLDI options");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-eps_arnoldi_delayed","Arnoldi with delayed reorthogonalization","EPSArnoldiSetDelayed",PETSC_FALSE,&arnoldi->delayed,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiSetDelayed_ARNOLDI"
PetscErrorCode EPSArnoldiSetDelayed_ARNOLDI(EPS eps,PetscTruth delayed)
{
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  arnoldi->delayed = delayed;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiSetDelayed"
/*@
   EPSArnoldiSetDelayed - Activates or deactivates delayed reorthogonalization 
   in the Arnoldi iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  delayed - boolean flag

   Options Database Key:
.  -eps_arnoldi_delayed - Activates delayed reorthogonalization in Arnoldi
   
   Note:
   Delayed reorthogonalization is an aggressive optimization for the Arnoldi
   eigensolver than may provide better scalability, but sometimes makes the
   solver converge less than the default algorithm.

   Level: advanced

.seealso: EPSArnoldiGetDelayed()
@*/
PetscErrorCode EPSArnoldiSetDelayed(EPS eps,PetscTruth delayed)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,delayed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiGetDelayed_ARNOLDI"
PetscErrorCode EPSArnoldiGetDelayed_ARNOLDI(EPS eps,PetscTruth *delayed)
{
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  *delayed = arnoldi->delayed;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiGetDelayed"
/*@C
   EPSArnoldiGetDelayed - Gets the type of reorthogonalization used during the Arnoldi
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  delayed - boolean flag indicating if delayed reorthogonalization has been enabled

   Level: advanced

.seealso: EPSArnoldiSetDelayed()
@*/
PetscErrorCode EPSArnoldiGetDelayed(EPS eps,PetscTruth *delayed)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,delayed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_ARNOLDI"
PetscErrorCode EPSView_ARNOLDI(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     isascii;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSARNOLDI",((PetscObject)viewer)->type_name);
  }
  if (arnoldi->delayed) {
    ierr = PetscViewerASCIIPrintf(viewer,"using delayed reorthogonalization\n");CHKERRQ(ierr);
  }  
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode EPSSolve_TS_ARNOLDI(EPS);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
PetscErrorCode EPSCreate_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARNOLDI    *arnoldi;
  
  PetscFunctionBegin;
  ierr = PetscNew(EPS_ARNOLDI,&arnoldi);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_ARNOLDI));
  eps->data                      = (void *)arnoldi;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->solvets              = EPSSolve_TS_ARNOLDI;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->setfromoptions       = EPSSetFromOptions_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_ARNOLDI;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  arnoldi->delayed               = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSArnoldiSetDelayed_C","EPSArnoldiSetDelayed_ARNOLDI",EPSArnoldiSetDelayed_ARNOLDI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSArnoldiGetDelayed_C","EPSArnoldiGetDelayed_ARNOLDI",EPSArnoldiGetDelayed_ARNOLDI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

int main(void)
{


    return 0;
}
