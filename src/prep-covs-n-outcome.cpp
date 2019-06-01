#include "prep-covs-n-outcome.h"

XY_dat::XY_dat(const arma::vec &y, const arma::mat &X, arma::vec w):
  y(y), X(X) {
#ifdef OUMU_DEBUG
  if(y.n_elem != X.n_rows or y.n_elem != w.n_elem)
    throw std::invalid_argument(
        "'y', 'w' and 'X' dimension do not match in 'XY_dat::XY_dat'");
#endif

  /* compute weighted versions */
  w.for_each( [](arma::mat::elem_type& val) { val = std::sqrt(val); } );
  c_sqw_y = y % w;
  s_sqw_X = X;
  s_sqw_X.each_col() %= w;

  /* compute weighted and centered versions */
  c_sqw_y  -= arma::mean(c_sqw_y);
  X_scales = arma::stddev(s_sqw_X, 0L, 0L);
  s_sqw_X.each_row() /= X_scales;
}
