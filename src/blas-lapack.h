#ifndef BLAS_LAPACK_H
#define BLAS_LAPACK_H
#include <R_ext/RS.h>	/* for F77_... */

#ifdef FC_LEN_T
# include <stddef.h> // for size_t if needed
# define FCLEN ,FC_LEN_T
# define FCONE ,(FC_LEN_T)1
#else
# define FCLEN
# define FCONE
#endif

extern "C" {
  void F77_NAME(dtrsm)(
      const char *side, const char *uplo,
      const char *transa, const char *diag,
      const int *m, const int *n, const double *alpha,
      const double *a, const int *lda,
      double *b, const int *ldb);
  void F77_NAME(dpotrs)(
      const char* uplo, const int* n,
      const int* nrhs, const double* a, const int* lda,
      double* b, const int* ldb, int* info);
  void F77_NAME(dgemv)(
      const char *trans, const int *m, const int *n,
      const double *alpha, const double *a, const int *lda,
      const double *x, const int *incx, const double *beta,
      double *y, const int *incy FCLEN);
}

#endif
