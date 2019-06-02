#ifndef OUMUAMUA_H
#define OUMUAMUA_H
#include <limits>
#include <memory>
#include <vector>
#include "arma.h"
#include  <cmath>

/* nodes in tree which represents the MARS model */
class cov_node {
public:
  /* covariate index */
  const unsigned cov_index;
  /* knot position. NaN means no hinge functions (just the linear term) */
  const double knot;
  const bool has_knots;
  /* sign for whether it is max(x - knot) or max(knot - x) */
  const double sign;
  /* depth starting from zero at root's children */
  const unsigned depth;
  /* list of child nodes */
  std::vector<std::unique_ptr<cov_node> > children;

  cov_node
    (const unsigned cov_index, const double knot, const double sign,
     const unsigned depth):
    cov_index(cov_index), knot(knot), has_knots(!std::isnan(knot)),
    sign(sign < 0. ? -1. : 1.), depth(depth)
    { }

  virtual ~cov_node() = default;
};

/* returned object with final model */
struct omua_res {
  /* childrens of the root node (first node with an intercept) */
  std::vector<std::unique_ptr<cov_node> > root_childrens;

  /* standard deviations of X */
  arma::vec X_scales;
};

/* Runs Multivariate Adaptive Regression Spline algorithm.
 *
 * Args:
 *   X: matrix with covariates.
 *   Y: vector with outcomes.
 *   W: vector with positive case weights.
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
 *   thresh: threshold for minimum R^2.
 */
omua_res omua
  (const arma::mat &X, const arma::vec &Y, const arma::vec &W,
   const double lambda, const unsigned endspan, const unsigned minspan,
   const unsigned degree, const unsigned nk, const double penalty,
   const unsigned trace, const double thresh);

#endif