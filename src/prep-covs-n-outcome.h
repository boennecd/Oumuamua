#ifndef PREP_H
#define PREP_H
#include "arma.h"

/* TODO: a bit overkill? */
struct XY_dat {
  /* class to hold covariates and outcomes */

  /* original outcomes and covariates */
  const arma::vec &y;
  const arma::mat &X;

  /* centered outcomes and scaled and centered covariates */
  arma::vec c_y;
  arma::mat sc_X;
  arma::rowvec X_means;
  arma::rowvec X_scales;
  double y_mean;

  /* Args:
   *   1: outcomes.
   *   2: covariates.
   */
  XY_dat(const arma::vec&, const arma::mat&);
};

void standardize(arma::mat&, const arma::rowvec&, const arma::rowvec&);

#endif
