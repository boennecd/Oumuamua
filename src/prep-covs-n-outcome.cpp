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
  sqw_y = y % w;
  sqw_X = X;
  sqw_X.each_col() %= w;

  /* compute weighted and centered versions */
  c_sqw_y  = sqw_y - arma::mean(sqw_y);

  X_means  = arma::mean(sqw_X, 0L);
  X_scales = arma::stddev(sqw_X, 0L, 0L);
  c_sqw_X  = sqw_X;
  c_sqw_X.each_row() -= X_means;
  c_sqw_X.each_row() /= X_scales;
}
