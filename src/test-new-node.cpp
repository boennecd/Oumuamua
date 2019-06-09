#ifdef IS_R_BUILD
#include "new-node.h"
#include <array>
#include <testthat.h>
#include <Rcpp.h>
#include "test-utils.h"
#include "miscellaneous.h"
#include "sort.h"
#include "knots.h"

static constexpr unsigned N = 25;

context("Testing 'get_new-node' and 'add_linear_term'") {
  auto get_data = [&]{
    Rcpp::RNGScope rngScope;
    arma::mat X(2L, N);
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
      arma::vec x = X.row(0).t();
      double lambda = 10.;

      auto brute = brute_solve(x, y, lambda);

      sort_keys idx(x);
      const arma::vec parent(N, arma::fill::ones);

      normal_equation old;
      const arma::mat B(0L, N);

      auto res = add_linear_term(old, x, y, parent, B, lambda, N, idx.order());

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
                x = X.row(1).t();
      arma::vec parent = X.row(0).t();
      set_hinge(parent, 0, 1, 1);
      double lambda = 10.;

      auto brute = brute_solve(x, y, lambda, parent);

      const arma::vec parent_cen = parent - mean(parent);
      normal_equation old(parent_cen.t() * parent_cen + lambda,
                          parent.t() * y);

      sort_keys idx(x);
      idx.subset(arma::find(parent.col(0) > 0));
      arma::mat B = parent_cen.t();

      auto res = add_linear_term(
        old, x, y, parent, B, lambda, N, idx.order());

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
        const double se = - arma::dot(coef, c + lambda * coef);
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
      arma::vec x = X.row(0).t();
      double lambda = 10.;
      sort_keys idx(x);
      arma::vec knots = x(idx.order());
      knots = knots.subvec(1L, N - 2L);

      /* brute force solution */
      auto brute = brute_solve(x, y, knots, lambda);

      /* implementation */

      arma::vec parent(N, arma::fill::ones);
      arma::mat B(0L, N);
      normal_equation eq;

      auto res = get_new_node(eq, x, y, parent, B, knots, lambda, N, false,
                              idx.order());

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
        arma::mat out(B_cen.n_cols, 4L);
        out.cols(0, 1) = B_cen.t();
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
        const double se = - arma::dot(coef, c + lambda * coef);
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
      arma::mat X = std::move(XY.first), B(2, X.n_cols);
      B.row(0) = X.row(0);
      B.row(1) = X.row(0);
      set_hinge(B, 0, 1, parent_knot, transpose);
      set_hinge(B, 1, -1, parent_knot, transpose);
      const arma::uword parent_idx = 1;

      arma::mat B_cen = B;
      center_cov(B_cen, 0, transpose);
      center_cov(B_cen, 1, transpose);

      /* get parent */
      const arma::vec parent = B.row(parent_idx).t();

      /* get outcome*/
      arma::vec y = std::move(XY.second);

      /* get x, subset that is active, and the knots */
      const arma::vec x_org = X.row(1).t();
      sort_keys idx(x_org);
      const arma::uvec keep = arma::find(parent > 0.);
      idx.subset(keep);
      const arma::vec x_active = x_org(idx.order());
      arma::vec knots = x_active;
      knots = knots.subvec(1L, knots.n_elem - 2L);

      /* set penalty parameter */
      const double lambda = 10.;

      /* brute force solution */
      auto brute = brute_solve(x_org, y, knots, lambda, parent, B_cen);

      /* implementation. First get the old solution */
      normal_equation eq;
      {
        arma::mat gram = B_cen * B_cen.t();
        gram.diag() += lambda;
        arma::vec k = B * y;
        eq.update(gram, k);
      }

      auto res = get_new_node(
        eq, x_org, y, parent, B_cen, knots, lambda, N, false, idx.order());

      expect_true(
        std::abs(brute.min_se_less_var - res.min_se_less_var) < 1e-8);
      expect_true(brute.knot == res.knot);

      /* also works with duplicates values */
      const arma::uword N2 = N * 2;
      arma::mat BD(2, N2);
      BD.row(0).subvec(0,  N - 1) = X.row(0);
      BD.row(0).subvec(N, N2 - 1) = X.row(0);
      BD.row(1).subvec(0,  N - 1) = X.row(1);
      BD.row(1).subvec(N, N2 - 1) = X.row(1);

      const arma::vec x_orgD = BD.row(1).t();

      set_hinge(BD, 0,  1, parent_knot, transpose);
      set_hinge(BD, 1, -1, parent_knot, transpose);

      B_cen = BD;
      center_cov(B_cen, 0, transpose);
      center_cov(B_cen, 1, transpose);

      /* get parent */
      const arma::vec parentD = BD.row(parent_idx).t();

      /* get outcome*/
      arma::vec yD(N2);
      yD.subvec(0,  N - 1) = y;
      yD.subvec(N, N2 - 1) = y;

      /* get x, subset that is active, and the knots */
      idx = sort_keys(x_orgD);
      const arma::uvec keepD = arma::find(parentD > 0.);
      idx.subset(keepD);

      arma::vec tmp = x_orgD(idx.order());
      knots = get_all_knots(tmp);

      /* brute force solution */
      brute = brute_solve(x_orgD, yD, knots, lambda, parentD, B_cen);

      /* implementation. First get the old solution */
      {
        arma::mat gram = B_cen * B_cen.t();
        gram.diag() += lambda;
        arma::vec k = BD * yD;
        eq.update_sub(gram, k);
      }

      res = get_new_node(
        eq, x_orgD, yD, parentD, B_cen, knots, lambda, N2, false, idx.order());

      expect_true(
        std::abs(brute.min_se_less_var - res.min_se_less_var) < 1e-8);
      expect_true(brute.knot == res.knot);
    }
  }

  test_that("'get_new_node' gives correct result with one hinge") {
    auto brute_solve = []
    (const arma::vec &x, const arma::vec &y_cen, const arma::vec &knots,
     const double lambda, const arma::vec &parent, const arma::mat &B_cen){
      const arma::mat B_base = ([&]{
        arma::mat out(B_cen.n_cols, 3);
        out.cols(0, 1) = B_cen.t();
        return out;
      })();

      new_node_res out;
      for(auto k : knots){
        arma::mat X = B_base;
        X.col(2) = x;
        set_hinge(X, 2L, 1., k);
        X.col(2) %= parent;
        arma::mat X_cen = X;
        center_cov(X_cen, 2L);

        arma::mat gram = X_cen.t() * X_cen;
        gram.diag() += lambda;
        const arma::vec c = X.t() * y_cen,
          coef = arma::solve(gram, c);
        const double se = - arma::dot(coef, c + lambda * coef);
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
      arma::mat X = std::move(XY.first), B(2, X.n_cols);
      B.row(0) = X.row(0);
      B.row(1) = X.row(0);
      set_hinge(B, 0, 1, parent_knot, transpose);
      set_hinge(B, 1, -1, parent_knot, transpose);
      const arma::uword parent_idx = 0;

      arma::mat B_cen = B;
      center_cov(B_cen, 0, transpose);
      center_cov(B_cen, 1, transpose);

      /* get outcome*/
      arma::vec y = std::move(XY.second);

      /* get x, subset that is active, and the knots */
      arma::vec x_org = X.row(0).t();
      sort_keys idx(x_org);
      arma::vec knots = x_org(idx.order());
      knots = knots.subvec(1L, knots.n_elem - 2L);

      /* set penalty parameter */
      double lambda = 10.;

      /* brute force solution */
      const arma::vec parent(x_org.n_elem, arma::fill::zeros);
      auto brute = brute_solve(x_org, y, knots, lambda, parent, B_cen);

      /* implementation. First get the old solution */
      normal_equation eq;
      {
        arma::mat gram = B_cen * B_cen.t();
        gram.diag() += lambda;
        arma::vec k = B * y;
        eq.update(gram, k);
      }

      auto res = get_new_node(
        eq, x_org, y, parent, B_cen, knots, lambda, N, true, idx.order());

      expect_true(
        std::abs(brute.min_se_less_var - res.min_se_less_var) < 1e-8);
      expect_true(brute.knot == res.knot);
    }
  }
}

#endif
