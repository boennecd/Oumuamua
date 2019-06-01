#ifdef IS_R_BUILD
#include "prep-covs-n-outcome.h"
#include <array>
#include <testthat.h>
#include "test-utils.h"

context("Testing 'XY_dat'") {
  test_that("Constructor of 'XY_dat' yields correct result") {
    /* R code
     X <- structure(c(0.71, 0.58, -0.53, 2.18, 1.42, -0.2, 0.57, 0.64,
     -0.67, 1.82, 0.47, 0.61), .Dim = c(6L, 2L))
     y <- c(1.01, 2.53, 1.68, 2.13, 0.35, 2.88)
     w <- c(0.98, 0.79, 0.72, 0.52, 0.57, 0.59)
     dput_2_cpp(X)
     dput_2_cpp(y)
     dput_2_cpp(w)

     wy <- sqrt(w) * y
     wX <- sqrt(w) * X

     dput_2_cpp(cwy <- scale(wy, scale = FALSE))
     dput_2_cpp(sds <- apply(wX, 2, sd))
     dput_2_cpp(cwX <- t(t(wX) / sds))
     */
    const arma::mat X = create_mat<6L, 2>(
      { 0.71, 0.58, -0.53, 2.18, 1.42, -0.2, 0.57, 0.64, -0.67, 1.82, 0.47, 0.61 });
    const arma::vec y = create_vec<6>({ 1.01, 2.53, 1.68, 2.13, 0.35, 2.88 }),
                    w = create_vec<6>({ 0.98, 0.79, 0.72, 0.52, 0.57, 0.59 });

    XY_dat XY(y, X, w);

    expect_true(is_all_equal(XY.X, X));
    expect_true(is_all_equal(XY.y, y));

    std::array<double, 6L> ey
    { -0.447895756572054, 0.800968442411011, -0.022217474297753, 0.0882200981778267, -1.18350053993536, 0.764425230216327 };
    std::array<double, 12L> eX
    {  0.933278966045799, 0.684512889809548, -0.597148312278801, 2.087364325492, 1.42352750734941, -0.203983994873437, 0.935856585728138, 0.943441393465553, -0.942893101410305, 2.17667940011148, 0.588513857054404, 0.777100606995665  };

    expect_true(is_all_aprx_equal(XY.s_sqw_X, eX));
    expect_true(is_all_aprx_equal(XY.c_sqw_y, ey));

    std::array<double, 2L> scales { 0.753112591273097, 0.602946241969155 };

    expect_true(is_all_aprx_equal(XY.X_scales, scales));
  }
}
#endif
