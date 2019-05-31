context("C++")
test_that("Catch unit tests pass", {
  set.seed(1L)
  expect_cpp_tests_pass("Oumuamua")
})
