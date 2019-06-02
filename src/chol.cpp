#include "chol.h"
#include "blas-lapack.h"

static constexpr double D_ONE = 1.;
static constexpr int I_ONE = 1L;
static constexpr char C_U = 'U', C_L = 'L', C_N = 'N', C_T = 'T';

using std::invalid_argument;

chol_decomp::chol_decomp(const arma::mat &X){
#ifdef OUMU_DEBUG
  if(X.n_rows != X.n_cols)
    throw invalid_argument("non-square matrix in 'chol_decomp::chol_decomp'");
#endif

  if(X.n_rows == 0L){
    chol_.resize(0L, 0L);
    return;
  }

  chol_ = arma::trimatu(arma::chol(X));
}

void chol_decomp::solve(arma::vec &out) const
{
#ifdef OUMU_DEBUG
  if(out.n_elem != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_decomp::solve'");
#endif

  int n = chol_.n_cols, info;
  F77_CALL(dpotrs)(&C_U, &n, &I_ONE, chol_.memptr(), &n, out.memptr(),
           &n, &info);
  if(info != 0)
    throw std::runtime_error("'dpotrs' failed with info " +
                             std::to_string(info));
}

arma::vec chol_decomp::solve(const arma::vec &x) const {
  arma::vec out = x;
  solve(out);
  return out;
}

void chol_decomp::update(const arma::mat &V) {
  const int n = chol_.n_cols, p = V.n_rows;
#ifdef OUMU_DEBUG
  if((n > 0L and p <= n) or (n == 0L and V.n_rows != V.n_cols))
    throw invalid_argument("dims do not match with 'chol_decomp::update'");
#endif

  if(n == 0L){
    /* no-prior decomposition to update */
    chol_ = arma::trimatu(arma::chol(V));
    return;
  }

  arma::mat new_chol(p, p, arma::fill::zeros);
  const arma::span old_span(0L, n - 1L), new_span(n, p - 1L);
  new_chol(old_span, old_span) = chol_;

  /* update upper right block */
  new_chol(old_span, new_span) = V.rows(old_span);
  const int nrhs = p - n;
  F77_CALL(dtrsm)(
      &C_L, &C_U, &C_T, &C_N, &n, &nrhs, &D_ONE, chol_.begin(), &n,
      new_chol.colptr(n), &p);

  /* update lower right block. TODO: do this inplace */
  new_chol(new_span, new_span) = arma::trimatu(arma::chol(
    V.rows(new_span.a, new_span.b) -
      new_chol(old_span, new_span).t() * new_chol(old_span, new_span)));

  chol_ = std::move(new_chol);
}
