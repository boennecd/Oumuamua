#ifndef NEW_NODE_H
#define NEW_NODE_H
#include "normal-eq.h"
#include <limits>
#include <memory>
#include <vector>

struct new_node_res {
  /* minimum squared error minus the sum of squared centered outcomes */
  double min_se_less_var = std::numeric_limits<double>::infinity();
  /* knot with smallest squared error */
  double knot = std::numeric_limits<double>::quiet_NaN();
};

inline double get_min_se_less_var
  (const normal_equation &eq, const double lambda){
  if(eq.n_elem() == 0L)
    return 0.;
  const arma::vec coef = eq.get_coef(),
                   rhs = eq.get_rhs();
  double out = 0;
  const double *r = rhs.begin();
  for(auto x : coef)
    out += x * (lambda * x + *r++);
  return -out;
}

/* Find best knot position with smallest squared error given a new predictor
 * and a given parent node. It takes an old normal equation Kx = z and
 * recursively updates V = | V_1^T V_2^T |^T and k in
 *
 *   | K     V_1 | | x_1 |   | z |
 *   |           | |     | = |   |
 *   | V_1^T V_2 | | x_2 |   | k |
 *
 * to find squared error at coefficients | x_1^T x_2^T |^T. All inputs related
 * to the observations are assumed to be ordered by the new covariate and only
 * *active* (non-zero) elements are included.
 *
 * Args:
 *   old_problem: "old" normal equation which we add two new equations to.
 *   We denote it by Kx = z.
 *   x: all covarite values in original order.
 *   y: all outcomes in original order.
 *   parent: values of parent node in original order.
 *   B: centered "old" design matrix.
 *   knots: interior knot positions to consider.
 *   lambda: L2 penalty on coefficients.
 *   N: total number of observations. TODO: can be infered from other
 *   arguments.
 *   one_hinge: true if one hinge should be added.
 *   indices: indices with order of *active* observations.
 */
new_node_res get_new_node
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const arma::vec &knots,
   const double lambda, const unsigned N, const bool one_hinge,
   const arma::uvec &indices);

struct add_linear_term_res {
  /* updated normal equation with covariate times parent */
  normal_equation new_eq;
  /* mean for covariate times the parent */
  const double x_parent_mean;
};

/* similar function to the one above but only adds an interaction between
 * the identity function and the parent node effect. */
add_linear_term_res add_linear_term
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const double lambda,
   const unsigned N, const arma::uvec &indices);

#endif
