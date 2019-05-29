#include "normal-eq.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"

context("Testing normal equation class") {
  test_that("Constructor and subsequent solve gives the correct result") {
    /* R code
    Q <- matrix(c(2, 1, 1, 3), 2L)
    dput_2_cpp(Q)
    x <- c(-0.74, -0.25)
    dput_2_cpp(x)

    z <- solve(Q, x)
    dput_2_cpp(z)
    */
    const arma::mat Q = create_mat<2L, 2>({ 2., 1., 1., 3. });
    const arma::vec x = create_vec<2L>({ -0.74, -0.25 });

    const normal_equation N(Q, x);
    std::array<double, 2L> expect { -0.394, 0.048 };
    arma::vec res = N.get_coef();
    expect_true(is_all_aprx_equal(res, expect));
  }

  test_that("gives correct result after 'update'") {
    /* R code
     Q <- matrix(c(1, 1:3,
     1, 3, 2:3,
     2, 2, 7, 3,
     3, 3, 3, 14), 4L)
     dput_2_cpp(Q)
     x <- c(-.1, .8, .3, -1)
     dput_2_cpp(x)
     dput_2_cpp(solve(Q[1:2, 1:2], x[1:2]))
     dput_2_cpp(solve(Q          , x     ))
    */
    normal_equation base;
    const arma::mat Q = create_mat<4L, 4L>(
    { 1., 1., 2., 3., 1., 3., 2., 3., 2., 2., 7., 3., 3., 3., 3., 14. });
    const arma::vec x = create_vec<4L>({ -0.1, 0.8, 0.3, -1. });

    {
      base.update(Q.submat(0L, 0L, 1L, 1L), x.subvec(0L, 1L));
      std::array<double, 2L> expect = { -0.55, 0.45 };

      arma::vec res = base.get_coef();
      expect_true(is_all_aprx_equal(res, expect));
    }

    {
      base.update(Q.submat(0L, 2L, 3L, 3L), x.subvec(2L, 3L));
      std::array<double, 4L> expect = { -0.383333333333333, 0.45, 0.0666666666666666, -0.1 } ;

      arma::vec res = base.get_coef();
      expect_true(is_all_aprx_equal(res, expect));
    }
  }
}
