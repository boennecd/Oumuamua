#ifdef IS_R_BUILD
#include "new-node.h"
#include <array>
#include <testthat.h>
#include <Rcpp.h>
#include "test-utils.h"
#include "miscellaneous.h"
#include "sort.h"

static constexpr unsigned N = 25;

context("Testing 'get_new-node' and 'add_linear_term'") {
  auto get_data = [&]{
    Rcpp::RNGScope rngScope;
    arma::mat X(N, 2L);
    for(auto &x : X)
      x = R::rnorm(1, 1);

    arma::mat y(N, 1L);
    for(auto &y : y)
      y = R::rnorm(0, 1);

    center_cov(y, 0L);

    return std::make_pair(std::move(X), std::move(y));
  };

  test_that("'add_linear_term' gives correct result without parent") {
    auto brute_solve = []
    (const arma::vec &x, const arma::vec &y_cen, const double lambda){
      /* dump version of doing this */
      const arma::mat x_cen = x - arma::mean(x);
      arma::vec V(1, 1);
      V(0, 0) = arma::dot(x_cen, x_cen) + lambda;
      arma::vec k(1);
      k.at(0) = arma::dot(x, y_cen);

      return normal_equation(V, k);
    };

    for(unsigned i = 0; i < 5L; ++i){
      auto XY = get_data();
      arma::mat X = std::move(XY.first);
      arma::vec y = std::move(XY.second);
      arma::vec x = X.col(0);
      double lambda = 10.;

      auto brute = brute_solve(x, y, lambda);

      sort_keys idx(x);
      x = x(idx.order());
      y = y(idx.order());
      const arma::vec parent(N, arma::fill::ones);

      normal_equation old;
      const arma::mat B(N, 0L);

      auto res = add_linear_term(old, x, y, parent, B, lambda, N);

      expect_true(is_all_aprx_equal(brute.get_rhs(), res.new_eq.get_rhs()));

      const arma::vec c1 = res.new_eq.get_coef(), c2 = brute.get_coef();
      expect_true(is_all_aprx_equal(c1, c2));
    }
  }

  test_that("'add_linear_term' gives correct result with parent") {
    auto brute_solve = []
    (const arma::vec &x, const arma::vec &y_cen, const double lambda,
     const arma::vec &parent){
      arma::mat B(N, 2L);
      B.col(0) = parent;
      B.col(1) = parent % x;

      arma::mat B_cen = B;
      center_cov(B_cen, 0);
      center_cov(B_cen, 1);

      arma::mat V = B_cen.t() * B_cen;
      V.diag() += lambda;

      arma::vec k = B.t() * y_cen;

      return normal_equation(V, k);
    };

    for(unsigned i = 0; i < 5L; ++i){
      auto XY = get_data();
      arma::mat X = std::move(XY.first);
      arma::vec y = std::move(XY.second),
                x = X.col(1);
      arma::vec parent = X.col(0);
      set_hinge(parent, 0, 1, 1);
      double lambda = 10.;

      auto brute = brute_solve(x, y, lambda, parent);

      arma::vec parent_cen = parent - mean(parent);
      normal_equation old(parent_cen.t() * parent_cen + lambda,
                          parent.t() * y);

      sort_keys idx(x);
      idx.subset(arma::find(parent.col(0) > 0));

      x = x(idx.order());
      y = y(idx.order());
      parent = parent(idx.order());
      parent_cen = parent_cen(idx.order());

      auto res = add_linear_term(old, x, y, parent, parent_cen, lambda, N);

      expect_true(is_all_aprx_equal(brute.get_rhs(), res.new_eq.get_rhs()));

      const arma::vec c1 = res.new_eq.get_coef(), c2 = brute.get_coef();
      expect_true(is_all_aprx_equal(c1, c2));
    }
  }

  test_that("'get_new_node' gives correct result without parent") {
    auto brute_solve = []
      (const arma::vec &x, const arma::vec &y_cen, const arma::vec &knots,
       const double lambda){
      new_node_res out;
      for(auto k : knots){
        arma::mat X(x.n_rows, 2L);
        X.col(0) = x;
        X.col(1) = x;
        set_hinge(X, 1L, 1., k);
        arma::mat X_cen = X;
        center_cov(X_cen, 0L);
        center_cov(X_cen, 1L);

        arma::mat gram = X_cen.t() * X_cen;
        gram.diag() += lambda;
        const arma::vec c = X.t() * y_cen,
                     coef = arma::solve(gram, c);
        const double se = - arma::dot(coef, c);
        if(se < out.min_se_less_var){
          out.min_se_less_var = se;
          out.knot = k;
        }
      }

      return out;
    };

    for(unsigned i = 0; i < 5L; ++i){
      auto XY = get_data();
      arma::mat X = std::move(XY.first);
      arma::vec y = std::move(XY.second);
      arma::vec x = X.col(0);
      double lambda = 10.;
      sort_keys idx(x);
      arma::vec knots = x(idx.order());
      knots = knots.subvec(1L, N - 2L);

      /* brute force solution */
      auto brute = brute_solve(x, y, knots, lambda);

      /* implementation */
      x = x(idx.order());
      y = y(idx.order());

      arma::vec parent(N, arma::fill::ones);
      arma::mat B(N, 0L);
      normal_equation eq;

      auto res = get_new_node(eq, x, y, parent, B, knots, lambda, N);

      expect_true(
        std::abs(brute.min_se_less_var - res.min_se_less_var) < 1e-8);
      expect_true(brute.knot == res.knot);
    }
  }

  test_that("'get_new_node' gives correct result with parent") {
    auto brute_solve = []
    (const arma::vec &x, const arma::vec &y_cen, const arma::vec &knots,
     const double lambda, const arma::vec &parent, const arma::mat &B_cen){
      const arma::mat B_base = ([&]{
        arma::mat out(B_cen.n_rows, 4L);
        out.cols(0, 1) = B_cen;
        out.col(2) = x % parent;
        center_cov(out, 2);
        return out;
      })();

      new_node_res out;
      for(auto k : knots){
        arma::mat X = B_base;
        X.col(3) = x;
        set_hinge(X, 3L, 1., k);
        X.col(3) %= parent;
        arma::mat X_cen = X;
        center_cov(X_cen, 3L);

        arma::mat gram = X_cen.t() * X_cen;
        gram.diag() += lambda;
        const arma::vec c = X.t() * y_cen,
          coef = arma::solve(gram, c);
        const double se = - arma::dot(coef, c);
        if(se < out.min_se_less_var){
          out.min_se_less_var = se;
          out.knot = k;
        }
      }

      return out;
    };

    for(unsigned i = 0; i < 5L; ++i){
      /* get previous design matrix, and centered design matrix */
      auto XY = get_data();
      const double parent_knot = 1.;
      arma::mat X = std::move(XY.first), B(X.n_rows, 2);
      B.col(0) = X.col(0);
      B.col(1) = X.col(0);
      set_hinge(B, 0, 1, parent_knot);
      set_hinge(B, 1, -1, parent_knot);
      const arma::uword parent_idx = 1;

      arma::mat B_cen = B;
      center_cov(B_cen, 0);
      center_cov(B_cen, 1);

      /* get parent */
      const arma::vec parent = B.col(parent_idx);

      /* get outcome*/
      arma::vec y = std::move(XY.second);

      /* get x, subset that is active, and the knots */
      const arma::vec x_org = X.col(1);
      sort_keys idx(x_org);
      const arma::uvec keep = arma::find(parent > 0.);
      idx.subset(keep);
      const arma::vec x_active = x_org(idx.order());
      arma::vec knots = x_active;
      knots = knots.subvec(1L, knots.n_elem - 2L);

      /* set penalty parameter */
      double lambda = 10.;

      /* brute force solution */
      auto brute = brute_solve(x_org, y, knots, lambda, parent, B_cen);

      /* implementation. First get the old solution */
      normal_equation eq;
      {
        arma::mat gram = B_cen.t() * B_cen;
        gram.diag() += lambda;
        arma::vec k = B.t() * y;
        eq.update(gram, k);
      }

      /* then get the active observations and call function */
      const arma::vec y_active = y(idx.order()),
                 parent_active = parent(idx.order());
      B_cen = B_cen.rows(idx.order());

      auto res = get_new_node(eq, x_active, y_active, parent_active,
                              B_cen, knots, lambda, N);

      expect_true(
        std::abs(brute.min_se_less_var - res.min_se_less_var) < 1e-8);
      expect_true(brute.knot == res.knot);
    }
  }
}

#endif
