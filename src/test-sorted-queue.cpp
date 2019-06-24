#ifdef IS_R_BUILD
#include <testthat.h>
#include "sorted_queue.h"
#include "test-utils.h"
#include <array>

namespace {
  struct dum {
    double a;
  };
}

inline double dum_key_func(const dum &x){
  return x.a;
}

context("Testing sorted_queue") {
  test_that("Gives corret order before and after sort") {
    sorted_queue<dum, dum_key_func> queue;
    queue.push_front(dum({ 1}));
    queue.push_front(dum({ 0}));
    queue.push_front(dum({-1}));
    queue.push_front(dum({ 2}));

    expect_true(queue[0].a ==  2);
    expect_true(queue[1].a == -1);
    expect_true(queue[2].a ==  0);
    expect_true(queue[3].a ==  1);

    queue.sort();
    expect_true(queue[0].a == -1);
    expect_true(queue[1].a ==  0);
    expect_true(queue[2].a ==  1);
    expect_true(queue[3].a ==  2);
  }
}

#endif
