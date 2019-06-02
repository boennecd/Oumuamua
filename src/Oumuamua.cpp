#include "Oumuamua.h"
#include "prep-covs-n-outcome.h"
#include "sort.h"
#include "new-node.h"
#include "knots.h"
#include "normal-eq.h"
#include <numeric>
#include <algorithm>
#include "miscellaneous.h"
#include "print.h"

using std::size_t;
static constexpr double no_knot = std::numeric_limits<double>::quiet_NaN();

namespace {
  /* class to hold data to pass around */
  struct problem_data {
    const arma::mat X;
    const arma::vec Y;
    const std::vector<sort_keys> keys;
    const unsigned endspan;
    const unsigned minspan;
    const double lambda;
    const unsigned N;
  };

  class extended_cov_node final : public cov_node {
    /* holds additional information than the base class */

    /* indices of covariate that can become children */
    mutable arma::uvec active_covs;

  public:
    /* sorted indices with of *active* (non-zero) observations */
    const sort_keys active_observations;
    /* term for all observations (active and non-active) in original order */
    const arma::vec x;

    extended_cov_node
      (const unsigned cov_index, const double knot, const double sign,
       const unsigned depth,
       const arma::uvec &active_covs, const sort_keys &cov_keys,
       const arma::uvec &active_subset, const arma::vec &x):
      cov_node(cov_index, knot, sign, depth),
      active_covs(active_covs),
      active_observations(([&]{
        sort_keys out = cov_keys;
        if(cov_keys.order().n_elem == active_subset.n_elem)
          return out;
        out.subset(active_subset);
        return out;
      })()), x(x) { }

    /* removes a covariate from the active list e.g., if one finds that all
     * covariate values are equal for the active observations */
    void remove_active_cov(const arma::uword idx) const {
      arma::uvec matches = arma::find(active_covs != idx);
      active_covs = active_covs(matches);
    }

    const arma::uvec& get_active_covs() const {
      return active_covs;
    }
  };
}

/* overload to avoid wrong order of arguments */
inline knot_res get_knots(const arma::vec &x, const problem_data &dat){
  return get_knots(x, dat.endspan, dat.minspan);
}

namespace {
  /* result type. 'all_equal' means no variation for the active covariates,
   * 'only_linear' means only a linear term should be included, and 'hinge'
   * means that two hinge functions should be included. */
  enum res_type { all_equal, only_linear, hinge };

  struct worker_res {
    res_type res = all_equal;
    /* minimum squared error minus the sum of squared centered outcomes */
    double min_se_less_var = std::numeric_limits<double>::infinity();
    /* knot with smallest squared error if a hinge function is used */
    double knot = std::numeric_limits<double>::quiet_NaN();
    /* covariate index */
    unsigned cov_index;
    /* pointer to parent node if the parent is not the root */
    const extended_cov_node * parent = nullptr;
  };

  /* worker class to be used for children of the root node */
  struct root_worker {
    const problem_data &dat;
    /* index of covariate to consider */
    const unsigned cov_index;
    /* should be a vector with one in all indices. Here to avoid many
     * constructions of such vectors */
    const arma::vec &parent;
    /* normal equation before updates */
    const normal_equation &old_eq;
    /* current centered design matrix */
    const arma::mat &cur_design_mat;

    worker_res operator()() const {
      worker_res out;
      out.cov_index = cov_index;

      /* get sorted covariate values and find knots */
      const sort_keys &key = dat.keys[cov_index];
      const arma::vec x = ([&]{
        arma::vec out = dat.X.col(cov_index);
        out = out(key.order());
        return out;
      })();
      auto knots = get_knots(x, dat);

      if(knots.n_unique < 2L){
        /* no variation in x */
        out.res = all_equal;
        return out;
      }

      const arma::vec y = dat.Y(key.order());
      const arma::mat B = cur_design_mat.rows(key.order());
      if(knots.n_unique < 3L or knots.knots.n_elem < 1L){
        /* could be a dummy. Include linear term */
        out.res = only_linear;
        auto lin_res = add_linear_term
          (old_eq, x, y, parent, B, dat.lambda, dat.N);
        out.min_se_less_var = get_min_se_less_var(lin_res.new_eq);
        return out;
      }

      /* find best knot position */
      {
        auto best_knot = get_new_node
          (old_eq, x, y, parent, B, knots, dat.lambda, dat.N);
        out.res = hinge;
        out.min_se_less_var = best_knot.min_se_less_var;
        out.knot = best_knot.knot;
      }

      return out;
    }
  };

  /* worker class to be used for children of the non-root node */
  struct non_root_worker {
    const problem_data &dat;
    /* index of covariate to consider */
    const unsigned cov_index;
    /* should be a vector with one in all indices. Here to avoid many
     * constructions of such vectors */
    const arma::vec &parent;
    /* normal equation before updates */
    const normal_equation &old_eq;
    /* pointer to parent node */
    const extended_cov_node * const parent_node;
    /* current centered design matrix */
    const arma::mat &cur_design_mat;

  public:
    worker_res operator()() const {
      worker_res out;
      out.parent = parent_node;
      out.cov_index = cov_index;

      /* get sorted covariate values and find knots */
      const sort_keys key = ([&]{
        sort_keys out = dat.keys[cov_index];
        out.subset(parent_node->active_observations);
        return out;
      })();

      if(key.order().n_elem < 3L){
        /* too few observations */
        out.res = all_equal;
        return out;
      }

      const arma::vec x = ([&]{
        arma::vec out = dat.X.col(cov_index);
        out = out(key.order());
        return out;
      })();
      auto knots = get_knots(x, dat);
      if(knots.n_unique < 2L){
        /* no variation in x */
        out.res = all_equal;
        return out;
      }

      /* TODO: re-order and copy is expensive here is all we want is to add a
       * slope */
      const arma::vec y = dat.Y(key.order()),
             /* TODO: avoid needing to have a full parent vector? */
             parent_use = parent(key.order());
      const arma::mat B = cur_design_mat.rows(key.order());
      if(knots.n_unique < 3L or knots.knots.n_elem < 1L){
        /* could be a dummy. Include linear term */
        out.res = only_linear;
        auto lin_res = add_linear_term
          (old_eq, x, y, parent_use, B, dat.lambda, dat.N);
        out.min_se_less_var = get_min_se_less_var(lin_res.new_eq);
        return out;
      }

      /* find best knot position */
      {
        auto best_knot = get_new_node
          (old_eq, x, y, parent_use, B, knots, dat.lambda, dat.N);
        out.res = hinge;
        out.min_se_less_var = best_knot.min_se_less_var;
        out.knot = best_knot.knot;
      }

      return out;
    }
  };

  /* GCV criterion with complexity function from Friedman (1992) */
  struct gcv {
    const double dN, Y_var, penalty;
    const unsigned &n_terms;

    struct gcv_output {
      const double gcv;
      const double Rsq;
    };

    gcv_output operator()
      (const double &min_se_less_var, const unsigned n_new_terms) const {
      /* TODO: account for L2 penalty? */
      const double ss_reg = (Y_var + min_se_less_var / dN),
                gcv_denum = 1. - get_complexity(n_new_terms) / dN;
      return { ss_reg / (gcv_denum * gcv_denum),
               1 - ss_reg / Y_var };
    }

    double get_complexity(const unsigned n_new_terms) const {
      const double n_total = n_terms + n_new_terms;
      return n_total  * (1. + penalty) + 1.;
    }
  };
}

template<class T>
void remove_first(T& container, const typename T::value_type value){
  auto idx = std::find(std::begin(container), std::end(container), value);
  if(idx != std::end(container))
    container.erase(idx);
}

omua_res omua
  (const arma::mat &X, const arma::vec &Y, const arma::vec &W,
   const double lambda, const unsigned endspan, const unsigned minspan,
   const unsigned degree, const unsigned nk, const double penalty,
   const unsigned trace, const double thresh){
  if(trace > 0)
    Oout << "Starting model estimation\n";

  /* number of observations */
  const size_t N = X.n_rows;
#ifdef OUMU_DEBUG
  {
    auto throw_err = [](const std::string &msg){
      throw std::invalid_argument("'omua_res': " + msg);
    };
    if(Y.n_elem != N or W.n_elem != N)
      throw_err("invalid 'Y' or W");
    if(degree < 1L)
      throw_err("invalid 'degree'");
    if(nk < 2L)
      throw_err("invalid 'nk'");
    if(endspan < 1L or minspan < 1L)
      throw_err("invalid 'endspan' or 'minspan'");
    if(penalty < 0.)
      throw_err("invalid 'penalty'");
    if(lambda < 0.)
      throw_err("invalid 'lambda'");
  }
#endif

  /* number of terms in the model */
  unsigned n_terms = 0L, n_covs = X.n_cols;
  /* output */
  omua_res out;

  /* set problem data object to use */
  const problem_data dat = ([&]{
    XY_dat dat(Y, X, W);
    out.X_scales = dat.X_scales.t();

    /* sorted indices for later */
    const std::vector<sort_keys> keys = ([&]{
      std::vector<sort_keys> out;
      out.reserve(n_covs);
      for(unsigned i = 0; i < n_covs; ++i)
        out.emplace_back(dat.s_sqw_X.col(i));

      return out;
    })();

    return problem_data
      { std::move(dat.s_sqw_X), std::move(dat.c_sqw_y), std::move(keys),
        endspan, minspan, lambda, X.n_rows };
  })();

  /* vector with root node's children */
  std::vector<std::unique_ptr<cov_node> > &root_childrens =
    out.root_childrens;

  /* vector with active covariates at root node (in-case the user passed
   * a covariate with all-equal values or we have dummies) */
  std::vector<arma::uword> active_root_covs(dat.X.n_cols);
  std::iota(active_root_covs.begin(), active_root_covs.end(), 0L);
  const std::vector<arma::uword> root_covs = active_root_covs;

  /* vector of ones used in 'root_worker' */
  const arma::vec root_parent(N, arma::fill::ones);

  /* final design matrix */
  arma::mat design_mat(N, nk, arma::fill::zeros);

  /* variance of Y */
  const double dN = (double)dat.N, Y_var = arma::dot(dat.Y, dat.Y) / dN;

  /* normal equation object we update */
  normal_equation eq;

  /* forward pass */
  std::vector<worker_res> results;
  results.reserve((int)(nk * 1.5 * X.n_cols)); /* TODO: find better value? */
  unsigned it = 0L;
  double old_Rsq = 0.;
  struct gcv gcv_comp { dN, Y_var, penalty, n_terms };
  while(n_terms + 2L <= nk){
    if(gcv_comp.get_complexity(2L) >= dN)
      break;
    it++;

    /* current centered design matrix */
    arma::mat cur_design_mat(design_mat.begin(), dat.N, n_terms, false);

    /* check root children */
    results.clear();
    for(auto i : active_root_covs){
      root_worker task { dat, i, root_parent, eq, cur_design_mat };
      results.push_back(task());
    }

    if(degree > 1L){
      /* TODO: iterate through children */
      throw std::runtime_error("not implemented");
    }

    /* find best new term */
    worker_res * best;
    double lowest_gcv = std::numeric_limits<double>::infinity();
    for(auto &r: results){
      const bool is_root_child = !r.parent;
      if(is_root_child){
        if(r.res == all_equal){
          /* remove the term so we do not call it again */
          remove_first(active_root_covs, r.cov_index);
          continue;
        }
      } else {
        throw std::runtime_error("not implemented");
      }

      const double gcv_val =
        gcv_comp(r.min_se_less_var, r.res == hinge ? 2L : 1L).gcv;
      if(gcv_val < lowest_gcv){
        lowest_gcv = gcv_val;
        best = &r;
      }
    }

    /* add new term */
    if(!best)
      /* TODO: fix */
      throw std::runtime_error("not implemented when no new term is found");

    const bool is_root_child = !best->parent;
    if(!is_root_child)
      throw std::runtime_error("not implemented");

    if(best->res == hinge){
      auto add_root_note = [&](const double sign){
        const unsigned cov_index = best->cov_index,
                       desg_indx = n_terms++;
        const double knot = best->knot;

        /* update design matrix. Wait with centering to later */
        design_mat.col(desg_indx) = dat.X.col(cov_index);
        set_hinge(design_mat, desg_indx, sign, knot);

        /* add node */
        {
          const arma::uvec active_subset = arma::find(
            design_mat.col(desg_indx) > 0);
          std::vector<arma::uword> active_covs = root_covs;
          remove_first(active_covs, cov_index);

          root_childrens.emplace_back(new extended_cov_node(
            cov_index, knot, sign, 0L, active_covs, dat.keys[cov_index],
            std::move(active_subset), design_mat.col(desg_indx)));
        }

        /* update normal equation */
        arma::vec k(1);
        k.at(0) = arma::dot(design_mat.col(desg_indx), dat.Y);
        center_cov(design_mat, desg_indx);
        arma::mat V =
          design_mat.cols(0, desg_indx).t() * design_mat.col(desg_indx);
        V(desg_indx, 0) += lambda;

        eq.update(V, k);
      };

      add_root_note(-1.);
      add_root_note(1.);

    } else if(best->res == only_linear){
      const unsigned cov_index = best->cov_index,
                     desg_indx = n_terms++;
      const double knot = no_knot;

      /* update design matrix. Wait with centering to later */
      design_mat.col(desg_indx) = dat.X.col(cov_index);

      /* add node */
      {
        /* just in case the covariate is sparse */
        const arma::uvec active_subset = arma::find(
          design_mat.col(desg_indx) != 0);
        /* remove this index from the root. Dont want to add it again */
        remove_first(active_root_covs, cov_index);

        root_childrens.emplace_back(new extended_cov_node(
            cov_index, knot, 0, 0L, active_root_covs, dat.keys[cov_index],
            std::move(active_subset), design_mat.col(desg_indx)));

        /* update normal equation */
        arma::vec k(1);
        k.at(0) = arma::dot(design_mat.col(desg_indx), dat.Y);
        center_cov(design_mat, desg_indx);
        arma::mat V =
          design_mat.cols(0, desg_indx).t() * design_mat.col(desg_indx);
        V(desg_indx, 0) += lambda;
        eq.update(V, k);
      }
    } else
      throw std::runtime_error("unsupported 'res_type'");

    auto stats = gcv_comp(get_min_se_less_var(eq), 0L);
    if(trace > 0){
      OPRINTF("Ended iteration %4d: GCV, R^2: %14.4f, %14.4f\n",
              it, stats.gcv, stats.Rsq);
      if(!best->parent){
        OPRINTF("Term at covariate %5d and knot at %14.4f\n",
                best->cov_index,
                best->knot * out.X_scales.at(best->cov_index));
      } else
        throw std::runtime_error("not implemented");
    }

    if(stats.Rsq < old_Rsq + thresh)
      break;
    old_Rsq = stats.Rsq;
  }

  /* backward pass. TODO: implement */

  return out;
}
