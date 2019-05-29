#ifndef NORMAL_EQ_H
#define NORMAL_EQ_H
#include "arma.h"
#include "chol.h"

class normal_equation {
  /* class that stores Cholesky decomposition of Gramian matrix, and vector
   * on right-hand-side. Mainly just a friendly wrapper to make life easier. */

  /* Cholesky decomposition of Gramian matrix */
  chol_decomp C = chol_decomp();
  /* vector on right-hand-side of normal equation */
  arma::vec z = arma::vec();

public:
  /* Setup normal equation for Kx = z */
  normal_equation(const arma::mat &K, const arma::vec &z):
  C(K), z(z) {
#ifdef OUMU_DEBUG
    if(K.n_rows != z.n_elem or K.n_rows != K.n_cols)
      throw std::invalid_argument(
          "invalid dimensions of 'K' and 'z' in 'normal_equation::normal_equation'");
#endif
  }
  normal_equation() = default;

  /* Update normal equation from Kx = z to
   *   | K     V_1 | | x_1 |   | z |
   *   |           | |     | = |   |
   *   | V_1^T V_2 | | x_2 |   | k |
   */
  void update(const arma::mat &V, const arma::vec &k) {
    const unsigned p = V.n_rows, n = C.get_decomp().n_rows;
#ifdef OUMU_DEBUG
    if(V.n_cols != k.n_elem or V.n_rows <= n)
      throw std::invalid_argument("Invalid 'V' or 'k' in 'normal_equation::update'");
#endif

    {
      const arma::span sold(0L, n - 1L), snew(n, p - 1L);
      arma::vec z_new(p);
      if(n > 0L)
        z_new(sold) = z;
      z_new(snew) = k;
      z = std::move(z_new);
    }
    C.update(V);
  }

  arma::vec get_coef() const {
    return C.solve(z);
  }
};

#endif
