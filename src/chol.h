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
  void update(const arma::mat &V){
    resize(V.n_rows);
    update_sub(V);
  }

  void update_sub(const arma::mat&);

  const arma::mat& get_decomp() const {
    return chol_;
  }

  void resize(const unsigned new_dim){
    if(new_dim < 1){
      chol_.set_size(0, 0);
      return;
    }

    arma::mat new_mat(new_dim, new_dim, arma::fill::zeros);
    if(chol_.n_cols > 0) {
      const arma::uword s = std::min(new_dim - 1, chol_.n_cols - 1);
      new_mat.submat(0, 0, s, s) = chol_.submat(0, 0, s, s);
    }
    chol_ = std::move(new_mat);
  }

  /* removes a column and row from the Cholesky decomposition */
  chol_decomp remove(const unsigned) const;
};

#endif
