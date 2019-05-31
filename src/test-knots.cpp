#ifdef IS_R_BUILD
#include "knots.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"

context("Testing knots functions") {
  test_that("'get_all_knots' yields correct result") {
    {
      const arma::vec x = create_vec<14L>(
        { 1.359, 1.359, 1.359, 1.231, 1.147, 0.564, 0.472, 0.456, 0.107, -0.251, -0.251, -0.251, -0.783, -1.027 });

      auto out = get_all_knots(x);
      std::array<double, 8> expect
        { 1.231, 1.147, 0.564, 0.472, 0.456, 0.107, -0.251, -0.783 };

      expect_true(is_all_aprx_equal(out.knots, expect));
      expect_true(out.n_unique == 10L);
    }
    {
      /* too few unique values */
      arma::vec x;
      x << 0 << 0 << 0;
      auto out = get_all_knots(x);
      expect_true(out.knots.n_elem == 0L);
      expect_true(out.n_unique == 1L);

      x << 1 << 1 << 0 << 0;
      out = get_all_knots(x);
      expect_true(out.knots.n_elem == 0L);
      expect_true(out.n_unique == 2L);
    }
  }
}

#endif
