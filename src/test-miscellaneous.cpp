#ifdef IS_R_BUILD
#include "miscellaneous.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"

/* dput_2_cpp(rnorm(5 * 2, 1)) */
static const arma::mat X = create_mat<2L, 5L>(
{ 0.607192070558016, 0.680007131451493, 0.720886697023441, 1.49418833126783, 0.822669517730394, 0.494042537885743, 2.34303882517041, 0.785420591453131, 0.820443469956613, 0.899809258786438 } );

context("Testing miscellaneous functions") {
  test_that("'center_cov' centers the covariates") {
    {
      arma::mat cp = X;
      center_cov(cp, 0L);
      expect_true(std::abs(arma::sum(cp.col(0))) < 1e-8);
      expect_true(std::abs(arma::sum(cp.col(1))) > 1e-8);
    }
    {
      arma::mat cp = X;
      center_cov(cp, 1L);
      expect_true(std::abs(arma::sum(cp.col(0))) > 1e-8);
      expect_true(std::abs(arma::sum(cp.col(1))) < 1e-8);
    }
  }
  test_that("'set_hinge' computes the correct hinge") {
    {
      const double knot = 1;
      arma::mat cp = X;
      set_hinge(cp, 0L, 1., knot);

      arma::vec z1 = cp.col(1), z2 = X.col(1);
      expect_true(is_all_aprx_equal(z1, z2));
      for(unsigned i = 0; i < cp.n_rows; ++i)
        expect_true(std::abs(
            cp.at(i) - std::max(X.at(i) - knot, 0.)) < 1e-8);
    }
    {
      const double knot = 1;
      arma::mat cp = X;
      set_hinge(cp, 1L, -1., knot);

      arma::vec z1 = cp.col(0), z2 = X.col(0);
      expect_true(is_all_aprx_equal(z1, z2));
      for(unsigned i = 0; i < cp.n_rows; ++i)
        expect_true(std::abs(
            cp.at(i, 1L) - std::max(knot - X.at(i, 1L), 0.)) < 1e-8);
    }
  }
}
#endif
