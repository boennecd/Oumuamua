#ifndef BLAS_LAPACK_H
#define BLAS_LAPACK_H
#include <R_ext/RS.h>	/* for F77_... */

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
}

#endif
