#ifndef GET_BASIS_H
#define GET_BASIS_H
#include "arma.h"
#include <algorithm>

/* set one of the columns in X to a hinge function using a knot, and a sign */
inline void set_hinge
  (arma::mat &X, const arma::uword idx, const double sign, const double knot){
#ifdef OUMU_DEBUG
  if(X.n_cols <= idx)
    throw std::invalid_argument("too large 'idx' in 'get_basis'");
#endif
  const double mult = sign < 0 ? -1. : 1.;
  X.col(idx).for_each( [&]( arma::mat::elem_type& val) {
    val = std::max(mult * (val - knot), 0.);
  });
}

inline void center_cov
  (arma::mat &X, const arma::uword idx){
#ifdef OUMU_DEBUG
  if(X.n_cols <= idx)
    throw std::invalid_argument("too large 'idx' in 'get_basis'");
#endif
  const double mean = arma::mean(X.unsafe_col(idx));
  X.col(idx) -= mean;
}

#endif
