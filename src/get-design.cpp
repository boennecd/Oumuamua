#include "get-design.h"
#include "prep-covs-n-outcome.h"
#include "miscellaneous.h"
#include <map>
#include <functional>
#include "print.h"
#include <algorithm>
#include "print.h"

namespace {
  class extended_node : public cov_node {
  public:
    /* term of this node */
    const arma::vec var;

    extended_node
      (const cov_node &source, const extended_node *parent,
       const arma::mat &X):
      cov_node(source.cov_index, source.knot, source.sign, source.depth,
               source.add_idx, parent),
       var(([&]{
        /* TODO: could be done quicker if parent is very sparse */
        arma::vec out = X.col(source.cov_index);
        set_hinge(out, 0, sign, knot);
        if(parent)
           out %= parent->var;
        return out;
      })()) { }
  };
}

arma::mat get_design(
    const std::vector<std::unique_ptr<cov_node> > &root_nodes,
    const arma::mat &X, const arma::vec &X_scales,
    const arma::vec &X_means, const arma::uvec &drop_order,
    const arma::uword n_vars, const bool with_penalty, const double lambda,
    const unsigned trace){
  if(trace > 1)
    Oout << "Creating design matrix\n";

  if(n_vars > drop_order.n_elem)
    throw std::invalid_argument("too large 'drop_from'");
  if(n_vars < 1)
    throw std::invalid_argument("No terms");
  /* center and scale X */
  const arma::mat std_X = ([&]{
    arma::mat out = X;
    standardize(out, X_means.t(), X_scales.t());
    return out;
  })();

  /* create simplified tree.
   * The small tree we need */
  std::vector<std::unique_ptr<extended_node> > small_tree;
  /* map from add index to node object */
  std::map<unsigned, extended_node*> cov_pointer;
  {
    /* create map of nodes and the index at which they are added in the
     * original tree */
    std::map<unsigned, const cov_node *> old_order;
    std::function<void(const cov_node &node)> add_old_node =
      [&](const cov_node &node) -> void {
        old_order[node.add_idx] = &node;
        for(auto &x : node.children)
          add_old_node(*x);
      };
    for(auto &x : root_nodes)
      add_old_node(*x);

    /* go through the terms we need, add the nodes we need to compute the
     * terms, and add create a map from a drop index to the node object */
    std::function<void(const arma::uword &idx)> add_new_node =
      [&](const arma::uword &idx) -> void {
        const cov_node &old_node = *old_order[idx];
#ifdef OUMU_DEBUG
        if(idx != old_node.add_idx)
          throw std::runtime_error("'idx' and 'old_node.add_idx' do not match");
#endif
        if(cov_pointer.find(idx) != cov_pointer.end())
          return;

        /* add this node in the new tree */
        if(old_node.parent){
          /* make sure the parent is added */
          add_new_node(old_node.parent->add_idx);

          /* then add this node to the new tree */
          extended_node * const parent =
            cov_pointer.find(old_node.parent->add_idx)->second;
          parent->children.emplace_back(
              new extended_node(old_node, parent, std_X));
          cov_pointer[idx] = (extended_node*)parent->children.back().get();

        } else {
          small_tree.emplace_back(new extended_node(old_node, nullptr, std_X));
          cov_pointer[idx] = small_tree.back().get();

        }
      };

    for(auto idx = drop_order.end() - n_vars; idx != drop_order.end(); ++idx)
      add_new_node(*idx);
  }

  /* form design matrix */
  arma::mat out(X.n_rows + with_penalty * n_vars, n_vars + 1);
  if(with_penalty)
    out.rows(X.n_rows, out.n_rows - 1).zeros();

  const arma::span covar_idx(0, X.n_rows - 1);
  out(covar_idx, arma::span(0, 0)).fill(1.);
  unsigned i = 1;
  for(auto idx = drop_order.end() - n_vars; idx != drop_order.end();
      ++idx, ++i){
    const extended_node &term_node = *cov_pointer[*idx];
    std::copy(term_node.var.begin(), term_node.var.end(),
              out.colptr(i));
    if(trace > 1)
      OPRINTF("Adding term (covariate idx, term index, knot, sign):  %5d %5d %14.4f %3d\n",
              term_node.cov_index, term_node.add_idx,
              term_node.knot * X_scales.at(term_node.cov_index) +
                X_means.at(term_node.cov_index), (int)term_node.sign);
  }

  if(with_penalty){
    const double lambda_sqrt = std::sqrt(lambda);
    for(unsigned j = 0; j < n_vars; ++j)
      out(X.n_rows + j, j + 1) = lambda_sqrt;
  }

  return out;
}
