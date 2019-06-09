context("Test with model with interactions")

interact_sim <- function(N, p){
  p <- max(5L, p)
  x <- matrix(runif(N * p), nc = p)
  y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 1/2)^2 +
    10 * x[, 4] + 5 * x[, 5] + rnorm(N)
  data.frame(y = y, x)
}
set.seed(6)
dat <- interact_sim(200, 5)

test_that("Get the same with model with interactions", {
  #####
  # check design matrix
  fit <- oumua(y ~ ., dat, control = oumua.control(
    nk = 30L, penalty = 2L, lambda = 10, endspan = 1L, minspan = 1L,
    degree = 4L))
  X <- fit$X[, -1]
  vnames <- colnames(X)
  vnames <- gsub(
    "(h\\()([^:]+)(\\))", "pmax(\\2, 0)", vnames, perl = TRUE)
  vnames <- gsub(":", "*", vnames)
  X_expect <- sapply(vnames, function(x) eval(parse(text = x), dat))
  # tolerance is high due to rounding
  expect_equal(X_expect, X, check.attributes = FALSE, tolerance = 1e-3)

  # test prediction
  yhat <- drop(fit$X %*% coef(fit))
  expect_equal(yhat, predict(fit, newdata = dat))
  expect_equal(yhat[1:10], predict(fit, newdata = dat[1:10, ]))
  expect_known_value(yhat, "interact-plain-yhat.RDS")

  # test output
  expect_s3_class(fit, "oumua")
  expect_known_value(fit[
    c("coefficients", "backward_stats")], "interact-plain-coef.RDS")

  #####
  # with weights
  expect_true(FALSE)

  #####
  # with different minspan and endspan
  fit <- oumua(y ~ ., dat, control = oumua.control(
    nk = 30L, penalty = 2L, lambda = 10, endspan = 30L, minspan = 30L,
    degree = 4L))
  expect_s3_class(fit, "oumua")
  expect_known_value(fit[
    c("coefficients", "backward_stats")], "interact-plain-large-span.RDS")
})

test_that("Get the same with  model with interactions and dummies", {
  fac <- gl(3, ceiling(nrow(dat) / 3))[1:nrow(dat)]
  dat$y <- dat$y +
    seq(-2, 2, length.out = length(levels(fac)))[as.integer(fac)]
  dat$fac <- fac

  fit <- oumua(y ~ ., dat, control = oumua.control(
    nk = 15L, penalty = 2, lambda = 10, endspan = 1L, minspan = 1L,
    degree = 3L))
  expect_s3_class(fit, "oumua")
  expect_known_value(fit[
    c("coefficients", "backward_stats")], "interact-dummy-coef.RDS")
})
