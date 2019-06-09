context("Test with additive model")

additiv_sim <- function(N, p){
  p <- max(5L, p)
  x <- matrix(runif(N * p), nc = p)
  y <- .1 * exp(4 * x[, 1]) + 4 / (1 + exp(-20 * (x[, 2] - 1/2))) +
    3 * x[, 3] + 2 * x[, 4] + x[, 5] + rnorm(N)
  data.frame(y = y, x)
}
set.seed(1)
dat <- additiv_sim(100, 5)

test_that("Get the same with additive model", {
  #####
  # check design matrix
  fit <- oumua(y ~ ., dat, control = oumua.control(
    nk = 15L, penalty = 2L, lambda = 10, endspan = 1L, minspan = 1L))
  X <- fit$X[, -1]
  vnames <- colnames(X)
  vnames <- gsub("(h\\()(.+)(\\))$", "pmax(\\2, 0)", vnames)
  X_expect <- sapply(vnames, function(x) eval(parse(text = x), dat))
  # tolerance is high due to rounding
  expect_equal(X_expect, X, check.attributes = FALSE, tolerance = 1e-3)

  # test prediction
  yhat <- drop(fit$X %*% coef(fit))
  expect_equal(yhat, predict(fit, newdata = dat))
  expect_equal(yhat[1:10], predict(fit, newdata = dat[1:10, ]))
  expect_known_value(yhat, "additive-plain-yhat.RDS")

  # test output
  expect_s3_class(fit, "oumua")
  expect_known_value(fit[
    c("coefficients", "backward_stats")], "additive-plain-coef.RDS")

  #####
  # with weights
  expect_true(FALSE)

  #####
  # with different penalty
  expect_true(FALSE)

  #####
  # with different minspan and endspan
  fit <- oumua(y ~ ., dat, control = oumua.control(
    nk = 15L, penalty = 2L, lambda = 10, endspan = 30L, minspan = 30L))
  expect_s3_class(fit, "oumua")
  expect_known_value(fit[
    c("coefficients", "backward_stats")], "additive-plain-large-span.RDS")
})

test_that("Get the same with additive model with dummies", {
  expect_true(FALSE)
})
