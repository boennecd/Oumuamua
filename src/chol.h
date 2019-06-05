#ifndef CHOL_H
#define CHOL
#include "arma.h"

class chol_decomp {
private:
  /* upper triangular matrix R */
  arma::mat chol_ = arma::mat();

public:
  /* computes R in the decomposition X = R^TR */
  chol_decomp() = default;
  chol_decomp(const arma::mat&);

  /* Return X^{-1}Z */
  void solve(arma::vec&) const;
  arma::vec solve(const arma::vec&) const;

  /* Update the Cholesky decomposition from X = R^TR to K = C^TC where
   *   K = | X     V_1 |
   *       | V_1^T V_2 |
   *
   * given V = | V_1^T V_2^T |^T
   */
  void update(const arma::mat&);

  const arma::mat& get_decomp() const {
    return chol_;
  }

  /* removes a column and row from the Cholesky decomposition */
  chol_decomp remove(const unsigned) const;
};

#endif
