#ifndef GET_DESIGN_H
#define GET_DESIGN_H
#include "Oumuamua.h"

/* returns the design matrix using a fixed number of terms */
arma::mat get_design(
    const std::vector<std::unique_ptr<cov_node> > &root_nodes,
    const arma::mat &X, const arma::vec &X_scales,
    const arma::vec &X_means, const arma::uvec &drop_order,
    const arma::uword n_vars, const bool with_penalty, const double lambda,
    const unsigned trace);

#endif
