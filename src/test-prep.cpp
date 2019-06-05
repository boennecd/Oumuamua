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

     wy <- w * y
     dput_2_cpp(scale(wy, scale = FALSE))
     dput_2_cpp(X <- scale(X))
     dput_2_cpp(attr(X, "scaled:center"))
     dput_2_cpp(attr(X, "scaled:scale"))
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
    {  0.0165906345730231, -0.112816315096557, -1.2177525776599, 1.47988460391366, 0.723351667383809, -0.88925801311404, -0.00422209362233656, 0.0844418724467292, -1.5748409211315, 1.57906301475384, -0.13088490229243, 0.046443029845701 };

    expect_true(is_all_aprx_equal(XY.sc_X, eX));
    expect_true(is_all_aprx_equal(XY.c_W_y, ey));

    std::array<double, 2L> scales { 1.00458283215804, 0.78949773062794 };
    expect_true(is_all_aprx_equal(XY.X_scales, scales));
    std::array<double, 2L> mean { 0.693333333333333, 0.573333333333333 };
    expect_true(is_all_aprx_equal(XY.X_means, mean));
  }
}
#endif
