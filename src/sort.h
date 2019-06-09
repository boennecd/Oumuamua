#ifndef SORT_H
#define SORT_H
#include "arma.h"
#include <numeric>
#include <set>
#include <algorithm>

class sort_keys {
  using index_pairs = std::map<std::size_t, std::size_t>;
  /* given a vector find ordered indices. A decreasing order is used. */
  arma::uvec order_vec;

public:
  sort_keys(const arma::vec &x) {
    order_vec = arma::sort_index(x, "descend");
  }

  /* map from unordered (subset) --> ordered (subset) */
  const arma::uvec& order() const {
    return order_vec;
  }
  operator const arma::uvec&() const {
    return order_vec;
  }

  /* subsets the original indices to keep. E.g., sorted indices are
   *   5, 4, 2, 3, 1
   *
   * and keep is
   *   1, 3, 9
   *
   * then the result is
   *   3, 1
   *
   * Complexity:
   *   main burden in current implementation is forming one binary tree on the
   *   caller side  and a series of log(N) lookups.
   */
  void subset(const std::set<arma::uword> &keep_set){
    if(order_vec.n_elem < 1L)
      return;

    unsigned n_add = 0.;
    arma::uvec order_new(std::min((std::size_t)order_vec.n_elem, keep_set.size()));
    arma::uword *n = order_new.begin();
    const auto set_end = keep_set.end();
    for(auto old : order_vec)
      if(keep_set.find(old) != set_end){
        *n++ = old;
        n_add++;
      }

    order_new.resize(n_add);
    order_vec = std::move(order_new);
  }
  void subset(const arma::uvec &keep_vec){
    const std::set<arma::uword> keep_set(keep_vec.begin(), keep_vec.end());
    subset(keep_set);
  }
};

#endif
