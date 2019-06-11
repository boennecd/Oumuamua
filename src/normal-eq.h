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

  /* to avoid computation of sum of squares */
  struct half_solve_z_obj {
    arma::vec half_solve = arma::vec();
    double ss = 0.;

    half_solve_z_obj() = default;
    half_solve_z_obj(const arma::vec &z, const chol_decomp &C):
      half_solve(([&]{
        arma::vec out = z;
        C.solve_half(out, transpose);
        return out;
      })()), ss(arma::dot(half_solve, half_solve)) { }
  };

  bool is_half_solve_set = false;
  half_solve_z_obj half_solve_z_;

  /* returns R^{-T}z */
  const half_solve_z_obj& get_half_solve_z(){
    if(!is_half_solve_set){
      half_solve_z_ = half_solve_z_obj(z, C);
      is_half_solve_set = true;
    }

    return half_solve_z_;
  }

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
#ifdef OUMU_DEBUG
    const unsigned n = C.get_decomp().n_rows;
    if(V.n_cols != k.n_elem or V.n_rows <= n or V.n_rows - n != k.n_elem)
      throw std::invalid_argument("Invalid 'V' or 'k' in 'normal_equation::update'");
#endif
    resize(V.n_rows);
    update_sub(V, k);
    is_half_solve_set = false;
  }

  /* same as above but updates last p parts */
  void update_sub(const arma::mat &V, const arma::vec &k){
#ifdef OUMU_DEBUG
    if(V.n_cols != k.n_elem)
      throw std::invalid_argument("Invalid 'k' in 'normal_equation::update_sub'");
    if(V.n_rows != n_elem())
      throw std::invalid_argument(
          "Invalid 'V' in 'normal_equation::update_sub' (" +
            std::to_string(V.n_rows) + ", " + std::to_string(n_elem()) + ")");
#endif
    z.subvec(z.n_elem - V.n_cols, z.n_elem - 1) = k;
    C.update_sub(V);
    is_half_solve_set = false;
  }

  void resize(const unsigned new_dim){
    z.reshape(new_dim, 1);
    C.resize(new_dim);
    is_half_solve_set = false;
  }

  /* returns the solution to the normal equation */
  arma::vec get_coef() const {
    return C.solve(z);
  }

  /* returns -(z^TK^{-1}z + lambda |coef|_2)*/
  double get_RSS_diff(const double lambda) const {
    if(z.n_elem == 0)
      return 0.;
    arma::vec tmp = z;
    C.solve_half(tmp);
    double out = arma::dot(tmp, tmp);
    if(lambda == 0)
      return -out;

    C.solve_half(tmp, no_transpose);
    return -out - lambda * arma::dot(tmp, tmp);
  }

  /* same as above but by passing a one-dimensional k and single
   * column V = | V_1^T V_2 |^t and working memory matrix with dimension
   * equal to the number of rows of V and two columns.
   *
   * The majority of the computation time of the program is spend here so this
   * needs to be fast.
   */
  double get_RSS_diff
    (const double lambda, const double k, const arma::mat &V,
     arma::mat &work_mem){
#ifdef OUMU_DEBUG
    if(V.n_cols != 1 or V.n_rows != z.n_elem + 1)
      throw std::invalid_argument("'get_RSS_diff': invalid 'V'");
    if(V.n_rows != work_mem.n_rows or work_mem.n_cols < 2)
      throw std::invalid_argument("'get_RSS_diff': invalid 'work_mem'");
    if(V.n_rows == 1)
      throw std::runtime_error("not implemented with empty 'normal_equation'");
#endif
    const unsigned m = z.n_elem;
    const arma::span old_ele(0, m - 1);
    const half_solve_z_obj &z_half = get_half_solve_z();

    /* compute the new column of V */
    arma::vec C_new_col(work_mem.memptr() , m + 1, false),
              z_new    (work_mem.colptr(1), m + 1, false);
    C_new_col = V.col(0);
    C.solve_half(C_new_col, transpose);
    C_new_col.at(m) = std::sqrt(
      V.at(m, 0) - arma::dot(C_new_col(old_ele), C_new_col(old_ele)));
    z_new.at(m) = (k - arma::dot(C_new_col(old_ele), z_half.half_solve)) /
      C_new_col.at(m);

    double out = -(z_new.at(m) * z_new.at(m) + z_half.ss);
    if(lambda == 0)
      return out;

    /* back substitution to get the coefficients */
    z_new.at(m) /= C_new_col.at(m);
    z_new(old_ele) = z_half.half_solve - z_new.at(m) * C_new_col(old_ele);
    C.solve_half(z_new, no_transpose);
    out -= lambda * arma::dot(z_new, z_new);
    return out;
  };

  const arma::vec& get_rhs() const {
    return z;
  }

  unsigned n_elem() const {
    return z.n_elem;
  }

  /* removes a variable from the equation */
  normal_equation remove(const unsigned idx) const {
#ifdef OUMU_DEBUG
    if(idx >= C.get_decomp().n_cols)
      throw std::invalid_argument(
          "'normal_equation::remove': index out for range");
#endif
    normal_equation out;
    out.C = C.remove(idx);
    out.z.resize(z.n_elem - 1L);
          double *o = out.z.begin();
    const double *v = z.cbegin();
    for(unsigned j = 0; j < z.n_elem; ++j, ++v)
      if(j != idx)
        *o++ = *v;

    return out;
  }
};

#endif
