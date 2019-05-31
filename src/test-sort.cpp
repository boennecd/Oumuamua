#ifdef IS_R_BUILD
#include "sort.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"

/* dput_2_cpp(rnorm(10)) */
static const arma::vec x = create_vec<10L>(
  { 0.712666307051405, -0.0735644041263263, -0.0376341714670479, -0.681660478755657, -0.324270272246319, 0.0601604404345152, -0.588894486259664, 0.531496192632572, -1.51839408178679, 0.306557860789766 } );

context("Testing sort_keys") {
  test_that("Constructor yields correct result") {
    sort_keys keys(x);
    arma::vec z1 = arma::sort(x, "descend"), z2 = x(keys.order());

    expect_true(is_all_aprx_equal(z1, z2));
  }
  test_that("gets correct reuslt after subsetting"){
    sort_keys keys(x);
    arma::uvec keep;
    {
      keep << 4 << 0 << 9 << 6 << 7 << 5;

      keys.subset(keep);
      arma::vec
        z1 = arma::sort(x(keep), "descend"),
        z2 = x(keys.order());

      expect_true(is_all_aprx_equal(z1, z2));
    }
    {
      keep.empty();
      keep << 4 << 1 << 5; /* 1 was not in the previous list */
      arma::uvec keep_expect;
      keep_expect << 4 << 5;

      keys.subset(keep);
      arma::vec
        z1 = arma::sort(x(keep_expect), "descend"),
        z2 = x(keys.order());

      expect_true(is_all_aprx_equal(z1, z2));
    }
  }
}

#endif
