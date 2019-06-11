#ifndef CHOL_H
#define CHOL_H
#include "arma.h"
#include "miscellaneous.h"

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

  /* Returns R^{-T}Z or R^{-1}Z */
  void solve_half
    (arma::vec&, const transpose_arg trans = transpose) const;
  arma::vec solve_half
    (const arma::vec&, const transpose_arg trans = transpose) const;

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
