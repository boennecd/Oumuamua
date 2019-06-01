#ifndef PREP_H
#define PREP_H
#include "arma.h"

struct XY_dat {
  /* class to hold covariates and outcomes */

  /* original outcomes and covariates */
  const arma::vec &y;
  const arma::mat &X;

  /* centered weighted outcomes, scaled covariates, and scales */
  arma::vec c_sqw_y;
  arma::mat s_sqw_X;
  arma::rowvec X_scales;

  /* Args:
   *   1: outcomes.
   *   2: covariates.
   *   3: weights.
   */
  XY_dat(const arma::vec&, const arma::mat&, arma::vec);
};

#endif
