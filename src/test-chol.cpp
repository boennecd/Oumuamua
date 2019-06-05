#include "chol.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"
#include <numeric>

context("Testing chol decomposition") {
  test_that("Constructor and solve gives the correct result") {
    /* R code
     Q <- matrix(c(2, 1, 1, 3), 2L)
     dput_2_cpp(Q)
     x <- c(-0.74, -0.25)
     dput_2_cpp(x)

     C <- chol(Q)
     dput_2_cpp(C)

     z <- solve(Q, x)
     dput_2_cpp(z)
     */
    const arma::mat Q = create_mat<2L, 2>({ 2., 1., 1., 3. });
    const arma::vec x = create_vec<2L>({ -0.74, -0.25 });

    const chol_decomp C(Q);
    {
      std::array<double, 4L> expect
        { 1.4142135623731, 0., 0.707106781186547, 1.58113883008419 };
      expect_true(is_all_aprx_equal(C.get_decomp(), expect));
    }

    {
      arma::vec z = x;
      const arma::vec o = C.solve(x);
      C.solve(z);

      expect_true(is_all_equal(z, o));
      std::array<double, 2L> expect { -0.394, 0.048 };

      expect_true(is_all_aprx_equal(o, expect));
    }
  }

  test_that("'get_decomp' gives correct result") {
    /* R code
     Q <- matrix(c(1, 1:3,
     1, 3, 2:3,
     2, 2, 7, 3,
     3, 3, 3, 14), 4L)
     dput_2_cpp(Q)
     Q1 <- Q[1:2, 1:2]
     C1 <- chol(Q1)
     dput_2_cpp(C1)
     C <- chol(Q)
     dput_2_cpp(C)
     */
    chol_decomp base;
    const arma::mat Q = create_mat<4L, 4L>(
      { 1., 1., 2., 3., 1., 3., 2., 3., 2., 2., 7., 3., 3., 3., 3., 14. });

    {
      const arma::mat Q1 = Q.submat(0L, 0L, 1L, 1L);
      base.update(Q1);
      std::array<double, 4L> expect { 1., 0., 1., 1.4142135623731 };

      expect_true(is_all_aprx_equal(base.get_decomp(), expect));
    }

    {
      const arma::mat V = Q.submat(0L, 2L, 3L, 3L);
      base.update(V);
      std::array<double, 16L> expect
      { 1., 0., 0., 0., 1., 1.4142135623731, 0., 0., 2., 0., 1.73205080756888,
        0., 3., 0., -1.73205080756888, 1.41421356237309 };

      expect_true(is_all_aprx_equal(base.get_decomp(), expect));
    }
  }

  test_that("'remove_chol' gives the correct result") {
    constexpr unsigned N = 5L;
    const arma::mat X = create_mat<N, N>(
    { 63.7, 19.15, 11.58, 5.3, 9.72, 19.15, 48.92, -3.23, 2.66, 6.64, 11.58, -3.23, 37.22, 6.12, 2.61, 5.3, 2.66, 6.12, 44.77, -3.13, 9.72, 6.64, 2.61, -3.13, 38.88 } );

    const chol_decomp base_chol(X);

    for(unsigned i = 0; i < N; ++i){
      const arma::uvec keep = ([&]{
        arma::uvec out(N - 1);
        arma::uword *o = out.begin();
        for(arma::uword j = 0; j < N; ++j)
          if(j != i)
            *o++ = j;

        return out;
      })();

      const arma::mat X_sub = X.submat(keep, keep);
      const chol_decomp expect(X_sub), actual = base_chol.remove(i);
      expect_true(is_all_aprx_equal(expect.get_decomp(), actual.get_decomp()));

      const arma::mat X_back = actual.get_decomp().t() * actual.get_decomp();
      expect_true(is_all_aprx_equal(X_sub, X_back));
    }
  }
}
