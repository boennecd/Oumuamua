#include "prep-covs-n-outcome.h"

void standardize(
    arma::mat &X, const arma::rowvec &X_means, const arma::rowvec &X_scales){
  X.each_row() -= X_means;
  X.each_row() /= X_scales;
}

XY_dat::XY_dat(const arma::vec &y, const arma::mat &X):
  y(y), X(X) {
#ifdef OUMU_DEBUG
  if(y.n_elem != X.n_rows)
    throw std::invalid_argument(
        "'y', 'w' and 'X' dimension do not match in 'XY_dat::XY_dat'");
#endif

  /* compute weighted and centered versions */
  c_y = y;
  y_mean = arma::mean(c_y);
  c_y  -= y_mean;
  sc_X = X;
  X_scales = arma::stddev(sc_X, 0L, 0L);
  X_means = arma::mean(sc_X, 0);
  standardize(sc_X, X_means, X_scales);
}
