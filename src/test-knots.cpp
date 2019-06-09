#ifdef IS_R_BUILD
#include "knots.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"
#include <numeric>

context("Testing knots functions") {
  test_that("'get_all_knots' yields correct result") {
    {
      const arma::vec x = create_vec<14L>(
        { 1.359, 1.359, 1.359, 1.231, 1.147, 0.564, 0.472, 0.456, 0.107, -0.251, -0.251, -0.251, -0.783, -1.027 });
      arma::uvec order(x.n_elem);
      std::iota(order.begin(), order.end(), 0);

      arma::vec dummy;
      auto out = get_all_knots<false>(x, order, dummy);
      std::array<double, 8> expect
        { 1.231, 1.147, 0.564, 0.472, 0.456, 0.107, -0.251, -0.783 };

      expect_true(is_all_aprx_equal(out.knots, expect));
      expect_true(out.n_unique == 10L);
    }
    {
      /* too few unique values */
      arma::vec x, dummy;
      x << 0 << 0 << 0;
      arma::uvec order;
      order << 0 << 1 << 2;
      auto out = get_all_knots<false>(x, order, dummy);
      expect_true(out.knots.n_elem == 0L);
      expect_true(out.n_unique == 1L);

      x << 1 << 1 << 0 << 0;
      order << 0 << 1 << 2 << 3;
      out = get_all_knots<false>(x, order, dummy);
      expect_true(out.knots.n_elem == 0L);
      expect_true(out.n_unique == 2L);
    }
  }

  test_that("'get_all_knots' yields correct result") {
    arma::vec x, dummy;
    arma::uvec i;
    x << 1 << 1 << 1 << 1 << 1 << 1 << 1;
    i << 0 << 1 << 2 << 3 << 4 << 5 << 6;

    auto out = get_knots_w_span<false>(x, 2, 3, i, dummy);
    expect_true(out.knots.n_elem == 0);
    expect_true(out.n_unique == 1);

    x << 2 << 1 << 1 << 1 << 1 << 1 << 0;
    out = get_knots_w_span<false>(x, 2, 3, i, dummy);
    arma::vec expect;
    expect << 1;
    expect_true(is_all_equal(out.knots, expect));
    expect_true(out.n_unique == 3);

    x << 9 << 8 << 7 << 6 << 5 << 4 << 3 << 2 << 1 << 0;
    i << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9;
    out = get_knots_w_span<false>(x, 1, 3, i, dummy);
    expect << 8 << 5 << 2 << 1;
    expect_true(is_all_equal(out.knots, expect));
    expect_true(out.n_unique == 10);

    out = get_knots_w_span<false>(x, 2, 2, i, dummy);
    expect << 7 << 5 << 3 << 2;
    expect_true(is_all_equal(out.knots, expect));
    expect_true(out.n_unique == 10);

    out = get_knots_w_span<false>(x, 4, 4, i, dummy);
    expect << 5 << 4;
    expect_true(is_all_equal(out.knots, expect));
    expect_true(out.n_unique == 10);
  }
}

#endif
