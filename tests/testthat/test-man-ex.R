context("Testing help page examples")

test_that("testing help page examples yield the same result", {
  #####
  # oumua
  library(Oumuamua)
  data("mtcars")
  fit <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
              control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
                                      degree = 2, n_threads = 1))
  f2 <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
              control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
                                      degree = 2, n_threads = 2))
  expect_equal(coef(fit), coef(f2))
  expect_known_value(fit[!names(fit) %in% c("call", "terms")],
                     "oumua-help.RDS")

  #####
  # predict.oumua
  expect_equal(drop(fit$X[1:10, ] %*% coef(fit)),
               predict(fit, mtcars[1:10, ]))

  #####
  # summary.oumua
  sum <- summary(fit)
  expect_known_output(sum, print = TRUE, file = "summary.txt")

  #####
  # simulation ex
  N <- 1000
  p <- 10
  set.seed(1)
  x <- matrix(runif(N * (p + 5)), N)
  true_f <-
    sin(pi * (x[, 1] + x[, 2] + x[, 3]         )) +
    sin(pi * (         x[, 2] + x[, 3] + x[, 4]))
  y <- true_f + rnorm(N)
  dat <- data.frame(y = y, x)

  fit <- oumua(y ~ ., dat, control = oumua.control(
    lambda = 1, endspan = 5L, minspan = 10L, penalty = 3, n_threads = 1L,
    nk = 50L, degree = 3))
  # dput(mean((true_f - predict(fit, newdata = dat))^2))
  expect_equal(mean((true_f - predict(fit, newdata = dat))^2), 0.173499103054566)

  fit <- oumua(y ~ ., dat, control = oumua.control(
    lambda = 1, endspan = 5L, minspan = 10L, penalty = 3, n_threads = 1L,
    nk = 50L, K = 20L, degree = 3))

  # dput(mean((true_f - predict(fit, newdata = dat))^2))
  expect_equal(mean((true_f - predict(fit, newdata = dat))^2), 0.190094489136486)

  fit <- oumua(y ~ ., dat, control = oumua.control(
    lambda = 1, endspan = 5L, minspan = 10L, penalty = 3, n_threads = 1L,
    nk = 50L, K = 20L, degree = 3, n_save = 3L))

  # dput(mean((true_f - predict(fit, newdata = dat))^2))
  expect_equal(mean((true_f - predict(fit, newdata = dat))^2), 0.198533207622569)
})
