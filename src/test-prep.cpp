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

     dput_2_cpp(wy <- sqrt(w) * y)
     dput_2_cpp(wX <- sqrt(w) * X)

     dput_2_cpp(cwy <- scale(wy, scale = FALSE))
     dput_2_cpp(cwX <- scale(wX))
     dput_2_cpp(attr(cwX, "scaled:center"))
     dput_2_cpp(attr(cwX, "scaled:scale"))
     */
    const arma::mat X = create_mat<6L, 2>(
      { 0.71, 0.58, -0.53, 2.18, 1.42, -0.2, 0.57, 0.64, -0.67, 1.82, 0.47, 0.61 });
    const arma::vec y = create_vec<6>({ 1.01, 2.53, 1.68, 2.13, 0.35, 2.88 }),
                    w = create_vec<6>({ 0.98, 0.79, 0.72, 0.52, 0.57, 0.59 });

    XY_dat XY(y, X, w);

    expect_true(is_all_equal(XY.X, X));
    expect_true(is_all_equal(XY.y, y));

    {
      std::array<double, 6L> ey
      { 0.999848988597778, 2.24871318758084, 1.42552727087208, 1.53596484334766, 0.264244205234476, 2.21216997538616 };
      std::array<double, 12L> eX
      { 0.702864140499428, 0.515515276204304, -0.449719912834644, 1.5720203561023, 1.07207648980845, -0.153622914957372,
        0.564271211386865, 0.568844442708198, -0.568513852073984, 1.31242066426889, 0.354842218457725, 0.468549890619985 };

      expect_true(is_all_aprx_equal(XY.sqw_X, eX));
      expect_true(is_all_aprx_equal(XY.sqw_y, ey));
    }

    {
      std::array<double, 6L> ey
      { -0.447895756572054, 0.800968442411011, -0.022217474297753, 0.0882200981778267, -1.18350053993536, 0.764425230216327 };
      std::array<double, 12L> eX
      { 0.212020402455046, -0.0367456737812055, -1.31840687586955, 1.36610576190125, 0.702268943758657, -0.92524255846419,
        0.189406795403982, 0.196991603141397, -1.68934289173446, 1.43022960978733, -0.157935933269752, 0.0306508166715086 };

      expect_true(is_all_aprx_equal(XY.c_sqw_X, eX));
      expect_true(is_all_aprx_equal(XY.c_sqw_y, ey));

      std::array<double, 2L> cens   { 0.543188905803744, 0.450069095894613 };
      std::array<double, 2L> scales { 0.753112591273096, 0.602946241969155 };

      expect_true(is_all_aprx_equal(XY.X_means, cens));
      expect_true(is_all_aprx_equal(XY.X_scales, scales));
    }
  }
}
