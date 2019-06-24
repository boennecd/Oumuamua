#include "Oumuamua.h"
#include "print.h"
#include "get-design.h"
#include <functional>
#include <memory>
#include "profile.h"

using Rcpp::List;
using Rcpp::Named;

// [[Rcpp::export]]
Rcpp::List omua_to_R
  (const arma::mat &X, const arma::vec &Y, const double lambda,
   const unsigned endspan, const unsigned minspan, const unsigned degree,
   const unsigned nk, const double penalty, const unsigned trace,
   const double thresh, const unsigned n_threads, const unsigned K,
   const unsigned n_save){
#ifdef OUMU_PROF
  profiler profiler("omua_to_R");
#endif

  auto comp_out = omua(
    X, Y, lambda, endspan, minspan, degree, nk, penalty, trace, thresh,
    n_threads, K, n_save);

  Rcpp::List out;
  /* find the number of variables to use and create the design matrix */
  {
    const arma::vec &GCVs = comp_out.backward_stats[1];
    const unsigned n_vars = std::distance(
      std::min_element(GCVs.begin(), GCVs.end()), GCVs.end());
    if(trace > 0)
      Rprintf("Using %d terms\n", n_vars);
    out["X"] = get_design(
      comp_out.root_childrens, X, comp_out.X_scales, comp_out.X_means,
      comp_out.drop_order, n_vars, true, lambda, trace);
    out["n_vars"] = n_vars;
  }

  /* add remaining elements */
  out["X_scales"] = std::move(comp_out.X_scales);
  out["X_means"] = std::move(comp_out.X_means);
  out["drop_order"] = std::move(comp_out.drop_order);

  out["backward_stats"] = List::create(
     Named("Rsq") = std::move(comp_out.backward_stats[0]),
     Named("GCV") = std::move(comp_out.backward_stats[1]));

  {
    std::function<List(const cov_node&)> wrap_cov_node  =
      [&](const cov_node &node) -> List {
        const unsigned n_childs =  node.children.size();
        List children(n_childs);
        for(unsigned i = 0; i < n_childs; ++i)
          children[i] = wrap_cov_node(*node.children[i]);

        return List::create(
          Named("cov_index") = node.cov_index,
          Named("knot") = node.knot,
          Named("has_knot") = node.has_knot,
          Named("sign") = node.sign,
          Named("add_idx") = node.add_idx,
          Named("children") = std::move(children));
      };
     const unsigned n_nodes = comp_out.root_childrens.size();
     List root_childrens(n_nodes);
     for(unsigned i = 0; i < n_nodes; ++i)
       root_childrens[i] = wrap_cov_node(*comp_out.root_childrens[i]);

     out["root_childrens"] = std::move(root_childrens);
  }

  return out;
}


// [[Rcpp::export]]
arma::mat get_design_map_from_R
  (const List root_childrens, const arma::mat &X, const arma::vec &X_scales,
   const arma::vec &X_means, const arma::uvec &drop_order,
   const arma::uword n_vars){
  /* create tree */
  std::vector<std::unique_ptr<cov_node> > root_nodes;

  unsigned max_cov_index = 0; /* used for checking */
  std::function<
    std::unique_ptr<cov_node>(const List, const cov_node * const parent,
                              const unsigned)>
    create_node =
    [&](const List x, const cov_node * const parent, const unsigned depth){
    std::unique_ptr<cov_node> out(new cov_node(
      Rcpp::as<int>(x["cov_index"]), Rcpp::as<double>(x["knot"]),
      Rcpp::as<double>(x["sign"]), depth, Rcpp::as<int>(x["add_idx"]),
      parent));

    if(out->cov_index > max_cov_index)
      max_cov_index = out->cov_index;

    List children = Rcpp::as<List>(x["children"]);
    out->children.reserve(children.size());
    for(auto x : children)
      out->children.push_back(create_node(
          Rcpp::as<List>(x), out.get(), depth + 1));

    return out;
  };

  root_nodes.reserve(root_childrens.size());
  for(auto &x : root_childrens)
    root_nodes.push_back(create_node(Rcpp::as<List>(x), nullptr, 0));

  /* checks */
  if(X.n_cols <= max_cov_index)
    throw std::runtime_error("invalid 'X'");
  if(drop_order.n_elem <= max_cov_index)
    throw std::runtime_error("invalid 'drop_order'");
  if(X_scales.n_elem != X.n_cols or X_means.n_elem != X.n_cols)
    throw std::runtime_error("invalid scales or means");

  return get_design(
    root_nodes, X, X_scales, X_means, drop_order, n_vars, false,
    std::numeric_limits<double>::quiet_NaN(), false);
}
