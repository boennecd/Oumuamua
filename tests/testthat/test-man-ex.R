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
})
