#ifndef OUMUAMUA_H
#define OUMUAMUA_H
#include <limits>
#include <memory>
#include <vector>
#include <array>
#include "arma.h"
#include <cmath>
#include <map>

/* nodes in tree which represents the MARS model */
class cov_node {
public:
  /* covariate index */
  const unsigned cov_index;
  /* knot position. NaN means no hinge functions (just the linear term) */
  const double knot;
  const bool has_knot;
  /* sign for whether it is max(x - knot) or max(knot - x) */
  const double sign;
  /* depth starting from zero at root's children */
  const unsigned depth;
  /* list of child nodes */
  std::vector<std::unique_ptr<cov_node> > children;
  /* index at which the node was added */
  const unsigned add_idx;
  /* parent. Null pointer means that the parent is the root note */
  const cov_node * const parent;

  cov_node
    (const unsigned cov_index, const double knot, const double sign,
     const unsigned depth, const unsigned add_idx,
     const cov_node * const parent):
    cov_index(cov_index), knot(knot), has_knot(!std::isnan(knot)),
    sign(sign < 0. ? -1. : 1.), depth(depth), add_idx(add_idx), parent(parent)
    { }

  virtual ~cov_node() = default;
};

/* returned object with final model */
struct omua_res {
  /* childrens of the root node (first node with an intercept) */
  std::vector<std::unique_ptr<cov_node> > root_childrens;

  /* standard deviations and means of X */
  arma::vec X_scales;
  arma::vec X_means;
  double y_mean;

  /* vector with coefficients from backward pass */
  std::vector<arma::vec> coefs;

  /* vector with order terms are dropped in (first to last) */
  arma::uvec drop_order;

  /* map from index at which the term is added to the node object */
  std::map<unsigned, const cov_node*> order_add;

  /* first element is R^2s and second element is GCVs  */
  std::array<arma::vec, 2> backward_stats;
};

/* Runs Multivariate Adaptive Regression Spline algorithm.
 *
 * Args:
 *   X: matrix with covariates.
 *   Y: vector with outcomes.
 *   lambda: L2 penalty parameter.
 *   endspan: minmum number of observations between the first observation and
 *   the first knot and similarly for the last knot and the last observation.
 *   minspan: minmum number of observations between first knot to all
 *   "interior" knots and to the last knot.
 *   degree: maximum degree of interactions.
 *   nk: maximum number of terms.
 *   penalty: d parameter in Friedman (1991) used in cost complexity function.
 *   trace: integer where zero yields no information and higher values yields
 *   more information.
 *   thresh: threshold for minimum R^2 improvment.
 *   n_threads: number of threads to use.
 *   K: maximum number of nodes to consider in forward pass. See the fast MARS
 *   paper (Friedman, 1991).
 *   n_save: number of iteration to save the covariate index as suggested in
 *   the fast MARS paper (Friedman, 1991).
 */
omua_res omua
  (const arma::mat &X, const arma::vec &Y,
   const double lambda, const unsigned endspan, const unsigned minspan,
   const unsigned degree, const unsigned nk, const double penalty,
   const unsigned trace, const double thresh, const unsigned n_threads,
   const unsigned K, const unsigned n_save);

#endif
