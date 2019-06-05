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
  c_W_y = y % w;

  /* compute weighted and centered versions */
  y_mean = arma::mean(c_W_y);
  c_W_y  -= y_mean;
  sc_X = X;
  X_scales = arma::stddev(sc_X, 0L, 0L);
  X_means = arma::mean(sc_X, 0);
  sc_X.each_row() -= X_means;
  sc_X.each_row() /= X_scales;
}
