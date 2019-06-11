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

void chol_decomp::solve_half
  (arma::vec &out, const transpose_arg trans) const {
#ifdef OUMU_DEBUG
  if(out.n_elem < chol_.n_cols)
    throw invalid_argument("too small 'out' in 'chol_decomp::solve'");
#endif
  const char trans_arg = trans == transpose ? 'T' : 'N';
  const int m = chol_.n_cols, ldb = out.n_elem;
  F77_CALL(dtrsm)(
      &C_L, &C_U, &trans_arg, &C_N, &m, &I_ONE, &D_ONE, chol_.begin(), &m,
      out.memptr(), &ldb);
}

arma::vec chol_decomp::solve_half
  (const arma::vec &Z, const transpose_arg trans) const {
  arma::vec out = Z;
  solve_half(out);
  return out;
}

void chol_decomp::update_sub(const arma::mat &V){
  const int p = V.n_rows, n = p - V.n_cols;
#ifdef OUMU_DEBUG
  if(V.n_rows != chol_.n_rows)
    throw invalid_argument("dims do not match with 'chol_decomp::update_sub'");
#endif

  if(n == 0L){
    /* no-prior decomposition to update */
    chol_ = arma::trimatu(arma::chol(V));
    return;
  }

  /* update upper right block */
  const arma::span old_span(0L, n - 1L), new_span(n, p - 1L);
  chol_(old_span, new_span) = V.rows(old_span);
  const int nrhs = p - n;
  F77_CALL(dtrsm)(
      &C_L, &C_U, &C_T, &C_N, &n, &nrhs, &D_ONE, chol_.begin(), &p,
      chol_.colptr(n), &p);

  /* update lower right block. */
  if(p - n == 1L){
    chol_(n, n) = V(n, 0);
    double *xn = chol_.colptr(n) + n;
    const double *v = chol_.colptr(n);
    for(int i = 0; i < n; ++i, ++v)
      *xn -= *v * *v;
    chol_(n, n) = std::sqrt(chol_(n, n));

  } else {
    chol_(new_span, new_span) = V.rows(new_span.a, new_span.b);
    chol_(new_span, new_span) -=
      chol_(old_span, new_span).t() * chol_(old_span, new_span);
    chol_(new_span, new_span) = arma::chol(chol_(new_span, new_span));

  }
}

inline void given_zero
  (const unsigned i, const unsigned j, arma::mat &X){
  const double a = X(i, i), b = X(j, i);
  double c, s;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  if(-eps < b and b < eps){
    return;

  } else {
    if(std::abs(b) > std::abs(a)){
      const double tau = - a / b;
      s = 1 / std::sqrt(1 + tau * tau);
      c = s * tau;
    } else {
      const double tau = - b / a;
      c = 1 / std::sqrt(1 + tau * tau);
      s = c * tau;
    }
  }

  const unsigned N = X.n_cols;
  bool flip_sign = false;
  for(unsigned k = i; k < N; ++k){
    const double tau1 = X(i, k),
                 tau2 = X(j, k);
    const double ival = c * tau1 - s * tau2;
    if(k == i)
      flip_sign = ival < 0.;
    X(i, k) = flip_sign ? -ival : ival;
    X(j, k) = s * tau1 + c * tau2;
  }
}

chol_decomp chol_decomp::remove(const unsigned idx) const {
  chol_decomp out;
  if(chol_.n_cols < 2 and idx == 0)
    return out;
#ifdef OUMU_DEBUG
  if(idx >= chol_.n_cols)
    throw std::invalid_argument(
        "'chol_decomp::remove_chol': index out of range");
#endif

  /* copy old values*/
  const unsigned p = chol_.n_cols - 1, pm1 = p - 1;
  out.chol_.resize(chol_.n_rows, p);
  arma::mat &ch_out = out.chol_;
  if(idx == 0)
    ch_out = chol_.cols(1, p);
  else if(idx == p)
    ch_out = chol_.cols(0, p - 1);
  else {
    ch_out.cols(0  , idx - 1) = chol_.cols(0, idx - 1);
    ch_out.cols(idx, pm1    ) = chol_.cols(idx + 1, p);
  }

  /* apply Givens rotations */
  for(unsigned i = idx; i < p; ++i){
    given_zero(i, i + 1, ch_out);
    ch_out(i + 1, i) = 0.;
  }

  out.chol_ = out.chol_.rows(0, pm1);
  return out;
}
