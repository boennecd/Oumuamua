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
#include "thread_pool.h"
#include "sorted_queue.h"

using std::size_t;
using std::to_string;
static constexpr double no_knot = std::numeric_limits<double>::quiet_NaN();

template<class T>
void remove_first(T& container, const typename T::value_type value){
  auto idx = std::find(std::begin(container), std::end(container), value);
  if(idx != std::end(container))
    container.erase(idx);
}

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
    mutable std::vector<arma::uword> active_covs;

  public:
    /* term for all observations (active and non-active) in original order */
    const arma::vec x;

    extended_cov_node
      (const unsigned cov_index, const double knot, const double sign,
       const unsigned depth, const unsigned add_idx,
       const cov_node * const parent, const std::vector<arma::uword> &active_covs,
       const sort_keys &cov_keys, const arma::uvec &active_subset,
       const arma::vec &x):
      cov_node(cov_index, knot, sign, depth, add_idx, parent),
      active_covs(active_covs),  x(x) { }

    /* removes a covariate from the active list e.g., if one finds that all
     * covariate values are equal for the active observations */
    void remove_active_cov(const arma::uword idx) const {
      remove_first(active_covs, idx);
    }

    const std::vector<arma::uword>& get_active_covs() const {
      return active_covs;
    }
  };

  /* class to a) keep track of which covariate pairs to consider and b)
   * intermediaries in the computations related to the fast MARS paper
   * (Friedman 1993) */
  struct task {
    /* active covariate indices */
    const std::vector<arma::uword> * active_covs = nullptr;
    /* pointer to parent node. Null if the task's parent is the root node. */
    extended_cov_node * parent = nullptr;
    /* last computed delta R^2 */
    double rss_delta = -std::numeric_limits<double>::infinity();

    task(const std::vector<arma::uword> &active_covs):
      active_covs(&active_covs) { }
    task(extended_cov_node &parent):
      active_covs(&parent.get_active_covs()), parent(&parent) { }
  };
}

inline double task_key(const task &x){
  return -x.rss_delta;
}

/* overload to avoid wrong order of arguments */
inline knot_res get_knots(
    const arma::vec &x, const problem_data &dat, const arma::uvec &indices){
  arma::vec dummy;
  return get_knots(x, dat.endspan, dat.minspan, indices, dummy);
}
inline knot_res get_knots(
    const arma::vec &x, const problem_data &dat, const arma::uvec &indices,
    const arma::vec &parent){
  return get_knots(x, dat.endspan, dat.minspan, indices, parent);
}

namespace {
  /* result type. 'all_equal' means no variation for the active covariates,
   * 'only_linear' means only a linear term should be included, 'one_hinge'
   * implies only one hinge function (if two have already been added for the
   * variable), and 'hinge' means that two hinge functions should be included. */
  enum res_type { all_equal, only_linear, one_hinge, hinge };

  struct worker_res {
    res_type res = all_equal;
    /* minimum squared error minus the sum of squared centered outcomes */
    double min_se_less_var = std::numeric_limits<double>::infinity();
    /* knot with smallest squared error if a hinge function is used */
    double knot = std::numeric_limits<double>::quiet_NaN();
    /* covariate index */
    unsigned cov_index;
    /* pointer to parent node if the parent is not the root */
    extended_cov_node * parent = nullptr;
    /* task associated with the pair */
    task &t;

    worker_res(task &t): t(t) { }
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
    /* current root nodes */
    const std::vector<std::unique_ptr<cov_node> > &root_childrens;
    /* task associated with pair */
    task &t;

    worker_res operator()() const {
      worker_res out(t);
      out.cov_index = cov_index;

      /* get sorted covariate values and find knots */
      const sort_keys &key = dat.keys[cov_index];
      const arma::vec x = dat.X.row(cov_index).t();
      auto knots = get_knots(x, dat, key.order());

      if(knots.n_unique < 2L){
        /* no variation in x */
        out.res = all_equal;
        return out;
      }

      const arma::vec &y = dat.Y;
      const arma::mat &B = cur_design_mat;

      if(knots.n_unique < 3L or knots.knots.n_elem < 1L){
        /* could be a dummy. Include linear term */
        out.res = only_linear;
        auto lin_res = add_linear_term
          (old_eq, x, y, parent, B, dat.lambda, dat.N, key.order());
        out.min_se_less_var = lin_res.new_eq.get_RSS_diff(dat.lambda);
        return out;
      }

      /* find best knot position */
      {
        auto cov_idx_ptr = std::find_if(
          root_childrens.begin(), root_childrens.end(),
          [&](const std::unique_ptr<cov_node> &node){
            return node->cov_index == cov_index;
          });
        const bool use_one_hinge = cov_idx_ptr != root_childrens.end();

        auto best_knot = get_new_node
          (old_eq, x, y, parent, B, knots, dat.lambda, dat.N,
           use_one_hinge, key.order());

        out.res = use_one_hinge ? one_hinge : hinge;
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
    extended_cov_node * const parent_node;
    /* current centered design matrix */
    const arma::mat &cur_design_mat;
    /* task associated with pair */
    task &t;

  public:
    worker_res operator()() const {
      worker_res out(t);
      out.parent = parent_node;
      out.cov_index = cov_index;

      /* get sorted covariate values and find knots */
      const sort_keys &key = dat.keys[cov_index];

      if(key.order().n_elem < 3L){
        /* too few observations */
        out.res = all_equal;
        return out;
      }

      const arma::vec x = dat.X.row(cov_index).t();
      auto knots = get_knots(x, dat, key.order(), parent);
      if(knots.n_unique < 2L){
        /* no variation in x */
        out.res = all_equal;
        return out;
      }

      const arma::vec &y = dat.Y,
            &parent_use = parent;
      const arma::mat &B = cur_design_mat;
      if(knots.n_unique < 3L or knots.knots.n_elem < 1L){
        /* could be a dummy. Include linear term */
        out.res = only_linear;
        auto lin_res = add_linear_term
          (old_eq, x, y, parent_use, B, dat.lambda, dat.N, key.order());
        out.min_se_less_var = lin_res.new_eq.get_RSS_diff(dat.lambda);
        return out;
      }

      /* find best knot position */
      {
        const auto &children = parent_node->children;
        const bool use_one_hinge = std::find_if(
          children.begin(), children.end(),
          [&](const std::unique_ptr<cov_node> &node){
            return node->cov_index != cov_index;
          }) != children.end();

        auto best_knot = get_new_node
          (old_eq, x, y, parent_use, B, knots, dat.lambda, dat.N,
           use_one_hinge, key.order());
        out.res = use_one_hinge ? one_hinge : hinge;
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
      double gcv;
      double Rsq;
    };

    gcv_output operator()
      (const double &min_se_less_var, const unsigned n_new_terms) const {
      const double
                ss_reg = (Y_var + min_se_less_var / dN),
                gcv_denum = 1. - get_complexity(n_new_terms) / dN;
      return { ss_reg / (gcv_denum * gcv_denum), 1 - ss_reg / Y_var };
    }

    double get_complexity(const unsigned n_new_terms) const {
      const double n_total = n_terms + n_new_terms;
      return n_total  * (1. + penalty) + 1.;
    }
  };
}

/* prints node information */
void print_knot_info(
    const cov_node &res, const arma::vec &X_scales, const arma::vec &X_means,
    const bool parent_print = false){
  if(res.parent)
    print_knot_info(*res.parent, X_scales, X_means, true);

  const std::string msg =
    parent_print ?
    "Term(s) info -- parent (covariate idx, knot, sign):  %5d, %14.4f %3d\n" :
    "Term(s) info           (covariate idx, knot, sign):  %5d, %14.4f %3d\n";

  OPRINTF(msg.c_str(),
          res.cov_index,
          res.knot * X_scales.at(res.cov_index) +
            X_means.at(res.cov_index),
            (int)res.sign);
}

void print_knot_info(
    const worker_res &res, const arma::vec &X_scales, const arma::vec &X_means,
    const bool parent_print = false){
  if(res.parent)
    print_knot_info(*res.parent, X_scales, X_means, true);

  const std::string msg =
    parent_print ?
    "Term(s) info -- parent       (covariate idx, knot):  %5d, %14.4f\n" :
    "Term(s) info                 (covariate idx, knot):  %5d, %14.4f\n";
  OPRINTF(msg.c_str(),
          res.cov_index,
          res.knot * X_scales.at(res.cov_index) +
            X_means.at(res.cov_index));

}

/* create a map from the index that the node is added to a pointer to the node */
inline std::map<unsigned, const cov_node*> get_order_to_idx
  (const std::vector<std::unique_ptr<cov_node> > &root_childrens){
  std::map<unsigned, const cov_node*> out;

  struct add_obs{
    std::map<unsigned, const cov_node*> &out;
    const cov_node &x;

    void operator()() {
      out[x.add_idx] = &x;
      for(auto &z : x.children)
        (add_obs {out, *z})();
    }
  };

  for(auto &x : root_childrens)
    (add_obs {out, *x})();

  return out;
}

omua_res omua
  (const arma::mat &X, const arma::vec &Y,
   const double lambda, const unsigned endspan, const unsigned minspan,
   const unsigned degree, const unsigned nk, const double penalty,
   const unsigned trace, const double thresh, const unsigned n_threads,
   const unsigned K){
  if(trace > 0)
    Oout << "Starting model estimation\n";

  /* number of observations */
  const size_t N = X.n_rows;
#ifdef OUMU_DEBUG
  {
    auto throw_err = [](const std::string &msg){
      throw std::invalid_argument("'omua_res': " + msg);
    };
    if(Y.n_elem != N)
      throw_err("invalid 'Y'");
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
    if(K < 1)
      throw_err("invalid 'K'");
  }
#endif

  /* number of terms in the model */
  unsigned n_terms = 0L, n_covs = X.n_cols;
  /* output */
  omua_res out;

  thread_pool pool(n_threads);

  /* set problem data object to use */
  const problem_data dat = ([&]{
    XY_dat dat(Y, X);
    out.X_scales = dat.X_scales.t();
    out.X_means = dat.X_means.t();
    out.y_mean = dat.y_mean;

    /* sorted indices for later */
    const std::vector<sort_keys> keys = ([&]{
      std::vector<sort_keys> out;
      out.reserve(n_covs);
      for(unsigned i = 0; i < n_covs; ++i)
        out.emplace_back(dat.sc_X.col(i));

      return out;
    })();

    arma::inplace_trans(dat.sc_X);
    return problem_data
      { std::move(dat.sc_X), std::move(dat.c_y), std::move(keys),
        endspan, minspan, lambda, X.n_rows };
  })();

  /* vector with root node's children */
  std::vector<std::unique_ptr<cov_node> > &root_childrens =
    out.root_childrens;

  /* vector with active covariates at root node (in-case the user passed
   * a covariate with all-equal values or we have dummies) */
  std::vector<arma::uword> active_root_covs(dat.X.n_rows);
  std::iota(active_root_covs.begin(), active_root_covs.end(), 0L);
  const std::vector<arma::uword> root_covs = active_root_covs;

  /* vector of ones used in 'root_worker' */
  const arma::vec root_parent(N, arma::fill::ones);

  /* final design matrix */
  arma::mat design_mat(nk, N, arma::fill::zeros);

  /* variance of Y */
  const double dN = (double)dat.N, Y_var = arma::dot(dat.Y, dat.Y) / dN;

  /* normal equation object we update */
  normal_equation eq;

  /* forward pass */
  {
    sorted_queue<task, task_key> work_queue;
    work_queue.emplace_back(active_root_covs);

    if(trace > 0)
      Oout << "Running foward pass\n";

    std::vector<worker_res> results;
    std::vector<std::future<worker_res> > futures;
    {
      /* TODO: find better value? */
      const unsigned max_tasks = nk * X.n_cols;
      results.reserve(max_tasks);
      futures.reserve(max_tasks);
    }

    unsigned it = 0L;
    double old_Rsq = 0.;
    struct gcv gcv_comp { dN, Y_var, penalty, n_terms };

    while(n_terms + 2L <= nk){
      if(gcv_comp.get_complexity(2L) >= dN)
        break;

#ifdef IS_R_BUILD
      if(it % 8 == 0)
        Rcpp::checkUserInterrupt();
#endif

      it++;

      /* current centered design matrix.
       * TODO: avoid this and pass full matrix around */
      const arma::mat cur_design_mat = ([&]{
        if(n_terms == 0)
          return arma::mat(0, dat.N);
        return (arma::mat)design_mat.rows(0, n_terms - 1);
      })();

      /* sort work queue if large enough */
      if(work_queue.size() > K)
        work_queue.sort();

      results.clear();
      futures.clear();
      {
        unsigned k = 0;
        for(auto &t : work_queue){
          if(k++ >= K)
            break;

          t.rss_delta = std::numeric_limits<double>::infinity();
          if(!t.parent){
            /* root job */
            for(auto i : *t.active_covs){
              root_worker job { dat, i, root_parent, eq, cur_design_mat,
                                root_childrens, t };
              futures.push_back(pool.submit(std::move(job)));
            }
            continue;

          }

          for(auto i : *t.active_covs){
            non_root_worker task {
              dat, i, t.parent->x, eq, t.parent, cur_design_mat, t };
            futures.push_back(pool.submit(std::move(task)));
          }
        }
      }

      /* gather results */
      for(auto &f : futures)
        results.push_back(f.get());

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
          if(r.res == all_equal){
            /* remove the term so we do not call it again */
            r.parent->remove_active_cov(r.cov_index);
            continue;
          }
        }

        /* update rss_delta */
        if(r.min_se_less_var < r.t.rss_delta)
          r.t.rss_delta = r.min_se_less_var;

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

      if(best->res == hinge or best->res == one_hinge){
        auto add_root_node = [&](const double sign){
          const unsigned cov_index = best->cov_index,
                         desg_indx = n_terms++;
          const double knot = best->knot;

          /* add node */
          extended_cov_node *new_node;
          if(!best->parent){
            /* update design matrix. Wait with centering to later */
            design_mat.row(desg_indx) = dat.X.row(cov_index);
            set_hinge(design_mat, desg_indx, sign, knot, transpose);

            const arma::uvec active_subset = arma::find(
              design_mat.row(desg_indx) > 0);
            std::vector<arma::uword> active_covs = root_covs;
            remove_first(active_covs, cov_index);

            root_childrens.emplace_back(new extended_cov_node(
              cov_index, knot, sign, 0L, desg_indx, nullptr, active_covs,
              dat.keys[cov_index], std::move(active_subset),
              design_mat.row(desg_indx).t()));
            new_node = (extended_cov_node*)root_childrens.back().get();

          } else {
            extended_cov_node &parent_node = *best->parent;

            /* update design matrix. Wait with centering to later */
            design_mat.row(desg_indx) = dat.X.row(cov_index);
            set_hinge(design_mat, desg_indx, sign, knot, transpose);
            design_mat.row(desg_indx) %= parent_node.x.t();

            const arma::uvec active_subset = arma::find(
              design_mat.row(desg_indx) > 0);
            std::vector<arma::uword> active_covs = parent_node.get_active_covs();
            remove_first(active_covs, cov_index);

            parent_node.children.emplace_back(new extended_cov_node(
                cov_index, knot, sign, parent_node.depth + 1, desg_indx,
                &parent_node, active_covs, dat.keys[cov_index],
                std::move(active_subset), design_mat.row(desg_indx).t()));
            new_node = (extended_cov_node*)parent_node.children.back().get();

          }

          /* update normal equation */
          arma::vec k(1);
          k.at(0) = arma::dot(design_mat.row(desg_indx), dat.Y);
          center_cov(design_mat, desg_indx, transpose);
          arma::mat V =
            design_mat.rows(0, desg_indx) * design_mat.row(desg_indx).t();
          V(desg_indx, 0) += lambda;

          eq.update(V, k);

          /* update work queue */
          if(new_node->depth < degree - 1)
            work_queue.emplace_back(*new_node);
        };

        if(best->res == hinge)
          add_root_node(-1.);
        add_root_node(1.);

      } else if(best->res == only_linear){
        const unsigned cov_index = best->cov_index,
                       desg_indx = n_terms++;
        const double knot = no_knot;

        /* add node */
        extended_cov_node *new_node;
        if(!best->parent){
          /* update design matrix. Wait with centering to later */
          design_mat.row(desg_indx) = dat.X.row(cov_index);

          /* just in case the covariate is sparse */
          const arma::uvec active_subset = arma::find(
            design_mat.row(desg_indx) != 0);
          /* remove this index from the root. Dont want to add it again */
          remove_first(active_root_covs, cov_index);

          root_childrens.emplace_back(new extended_cov_node(
              cov_index, knot, 0, 0L, desg_indx, nullptr, active_root_covs,
              dat.keys[cov_index], std::move(active_subset),
              design_mat.row(desg_indx).t()));
          new_node = (extended_cov_node*)root_childrens.back().get();

        } else {
          extended_cov_node &parent_node = *best->parent;

          /* update design matrix. Wait with centering to later */
          design_mat.row(desg_indx) = dat.X.row(cov_index);
          design_mat.row(desg_indx) %= parent_node.x.t();

          const arma::uvec active_subset = arma::find(
            design_mat.row(desg_indx) != 0);
          parent_node.remove_active_cov(cov_index);

          parent_node.children.emplace_back(new extended_cov_node(
              cov_index, knot, 0, parent_node.depth + 1, desg_indx, nullptr,
              active_root_covs, dat.keys[cov_index], std::move(active_subset),
              design_mat.row(desg_indx).t()));
          new_node = (extended_cov_node*)parent_node.children.back().get();

        }

        /* update normal equation */
        arma::vec k(1);
        k.at(0) = arma::dot(design_mat.row(desg_indx), dat.Y);
        center_cov(design_mat, desg_indx, transpose);
        arma::mat V =
          design_mat.rows(0, desg_indx) * design_mat.row(desg_indx).t();
        V(desg_indx, 0) += lambda;
        eq.update(V, k);

        /* update work queue */
        if(new_node->depth < degree - 1)
          work_queue.emplace_back(*new_node);

      } else
        throw std::runtime_error("unsupported 'res_type'");

      auto stats = gcv_comp(eq.get_RSS_diff(dat.lambda), 0L);
      if(trace > 0){
        OPRINTF("Ended iteration %4d: GCV, R^2, # terms: %14.4f, %14.4f, %4d\n",
                it, stats.gcv, stats.Rsq, eq.n_elem());
        print_knot_info(*best, out.X_scales, out.X_means);
      }

      if(stats.Rsq < old_Rsq + thresh)
        break;
      old_Rsq = stats.Rsq;
    }
  }

  /* create map with index added and pointer to node */
  std::map<unsigned, const cov_node*>  &order_add
    = out.order_add;
  order_add = get_order_to_idx(root_childrens);

#ifdef OUMU_DEBUG
  {
    auto v_min = std::min_element(order_add.begin(), order_add.end()),
         v_max = std::max_element(order_add.begin(), order_add.end());
    if(v_min == order_add.end() or v_max == order_add.end())
      throw std::runtime_error("Invalid 'order_add' (nullpointers)");

    if(order_add.size() != n_terms or v_min->first != 0 or
         v_max->first != n_terms - 1L)
      throw std::runtime_error(
          "Invalid 'order_add' (" + to_string(order_add.size()) +
            ", " + to_string(v_min->first) + ", " + to_string(v_max->first));
  }
#endif

  /* backward pass */
  arma::uvec &drop_order = out.drop_order;
  arma::vec &R2sq = out.backward_stats[0], &GCVs = out.backward_stats[1];
#ifdef OUMU_DEBUG
  std::vector<arma::vec> &coefs = out.coefs;
#endif
  {
    if(trace > 0)
      Oout << "Running backward pass\n";

    drop_order.set_size(n_terms);
    R2sq.set_size(n_terms);
    GCVs.set_size(n_terms);

    /* normal equation which we will remove equations from */
    normal_equation working_model = eq;

    /* remaning indices */
    std::vector<unsigned> remaning(n_terms);
    std::iota(remaning.begin(), remaning.end(), 0);

    const unsigned n_terms_dummy = 0L;
    struct gcv gcv_comp { dN, Y_var, penalty, n_terms_dummy };
    auto get_gcv = [&](const normal_equation &mod){
      return gcv_comp(mod.get_RSS_diff(dat.lambda), mod.n_elem());
    };

    /* set values with all terms */
#ifdef OUMU_DEBUG
    coefs.reserve(n_terms);
    coefs.emplace_back(working_model.get_coef());
#endif
    {
      auto stats = get_gcv(working_model);
      R2sq.at(0) = stats.Rsq;
      GCVs.at(0) = stats.gcv;
    }

    unsigned i = 0;
    while(working_model.n_elem() > 0){
      if(working_model.n_elem() == 1L){
        drop_order[i] = remaning[0];
        break;
      }

      /* remove equations from normal equation and find the one with the
       * lowest GCV */
      double min_gcv = std::numeric_limits<double>::infinity();
      unsigned idx_to_drop = 0L;
#ifdef OUMU_DEBUG
      gcv::gcv_output max_stats {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
      };
#endif
      for(unsigned j = 0; j < working_model.n_elem(); ++j){
        normal_equation one_less = working_model.remove(j);
        auto stats = get_gcv(one_less);
        if(trace > 1){
          OPRINTF("j, term, GCV: %4d %4d %20.5f\n", j, remaning[j], stats.gcv);
          arma::vec t1(remaning.size()), t2 =  one_less.get_coef();
          const double *x = t2.begin();
          for(unsigned i = 0; i < remaning.size(); ++i)
            if(i != j)
              t1[i] = *x++;
            else
              t1[i] = 0.;
          Oout << "Coef: " << t1.t();

        }
        if(stats.gcv < min_gcv){
          min_gcv = stats.gcv;
#ifdef OUMU_DEBUG
          max_stats = stats;
#endif
          idx_to_drop = j;
        }
      }

      /* drop node */
      working_model = working_model.remove(idx_to_drop);
      drop_order[i++] = remaning[idx_to_drop];
      remaning.erase(remaning.begin() + idx_to_drop);
      auto new_stats = get_gcv(working_model);

      R2sq.at(i) = new_stats.Rsq;
      GCVs.at(i) = new_stats.gcv;

#ifdef OUMU_DEBUG
      coefs.emplace_back(working_model.get_coef());
      if(max_stats.gcv - new_stats.gcv  != 0)
        throw std::runtime_error("GCV do not match after removal");
#endif
      if(trace > 0){
        OPRINTF("Dropped term %4d: GCV, R^2, # coef: %14.4f, %14.4f, %4d\nRemaining terms indices ",
                drop_order[i - 1L], new_stats.gcv, new_stats.Rsq,
                working_model.n_elem());
        for(auto i : remaning)
          OPRINTF("%4d ", i);
        Oout << "\n";
        print_knot_info(
          *order_add.find(drop_order[i - 1])->second,
          out.X_scales, out.X_means);

      }
    }
  }

#ifdef OUMU_DEBUG
  {
    unsigned n_expect = n_terms;
    for(auto &x: coefs)
      if(x.n_elem != n_expect--)
        throw std::runtime_error(
            "Wrong size of coefficients (" + to_string(x.n_elem) +
              ", " + to_string(n_expect + 1) + ")");

    arma::uvec cp = drop_order;
    std::sort(cp.begin(), cp.end());
    for(auto i = cp.begin(); i + 1 != cp.end(); ++i)
      if(*i == *(i + 1))
        throw std::runtime_error("Duplicate values in 'drop_order'");
  }
#endif

  return out;
}
