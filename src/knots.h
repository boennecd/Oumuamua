#ifndef KNOTS_H
#define KNOTS_H
#include "arma.h"
#include <iterator>

using std::size_t;

struct knot_res {
  arma::vec knots;
  size_t n_unique;

  operator const arma::vec&() const {
    return knots;
  }
};

inline void get_knots_check_input
  (const arma::vec &x, const arma::uvec &order, const unsigned endspan,
   const unsigned minspan){
#ifdef OUMU_DEBUG
  if(x.n_elem < order.n_elem)
    throw std::invalid_argument("'get_all_knots': larger 'order'");
  arma::uvec tmp = arma::find(arma::diff(x(order)) > 0.);
  if(tmp.n_elem > 0L)
    throw std::invalid_argument("'get_all_knots': 'x' is not decreasing");
  if(endspan < 1)
    throw std::runtime_error("'get_knots_w_span': too small 'endspan'");
  if(minspan < 1)
    throw std::runtime_error("'get_knots_w_span': too small 'minspan'");
  if(order.n_elem < 3L)
    throw std::invalid_argument("'get_all_knots': too small 'x'");
#endif
}

/* vector and an order and returns all interior unique values. Returns and
 * empty vector if input has two or fewer unqiue values. */
template<bool has_parent>
knot_res get_all_knots(const arma::vec &x, const arma::uvec &order,
                       const arma::vec &parent){
  get_knots_check_input(x, order, 1, 1);

  arma::vec out(x.n_elem - 2L);
  size_t n_ele = 0L;
  double old = x[order[0]], dnew = 0, *o = out.begin();
  for(auto i = order.begin() + 1L; i != order.end(); old = dnew, ++i){
    if(has_parent and parent[*i] == 0.)
      continue;
    dnew = x[*i];
    if(dnew < old){
      *o++ = dnew;
      n_ele++;
      if(o == out.end())
        break;
    }
  }

  if(n_ele < 2L)
    return { arma::vec(), n_ele + 1L };

  return { out.subvec(0L, n_ele - 2L), n_ele + 1L };
}

template<bool has_parent>
knot_res get_knots_w_span
  (const arma::vec &x, const unsigned endspan, const unsigned minspan,
   const arma::uvec &order, const arma::vec &parent){
  get_knots_check_input(x, order, endspan, minspan);

  size_t n_unique = 0;
  if(order.n_elem <= 2 * endspan){
    /* just need to count unique values and return */
    double x_old = x[order[0]];
    n_unique++;
    for(auto i = order.begin() + 1; i != order.end(); ++i){
      if(has_parent and parent[*i] == 0)
        continue;

      double x_new = x[*i];
      if(x_new < x_old){
        n_unique++;
        x_old = x_new;
      }
    }

    return { arma::vec(), n_unique };
  }

  /* we first find the the first and last knot */
  const arma::uword *start_knot, *end_knot = nullptr;
  {
    /* start with first knot */
    auto i = order.begin();
    for(; has_parent and parent[*i] == 0 and i != order.end(); ++i);
    const double x_first = x[*i];

    double x_old = x[*i++];
    n_unique++;
    for(unsigned k = 1; k != endspan and i != order.end(); ++i, ++k){
      if(has_parent and parent[*i] == 0){
        k--;
        continue;
      }

      const double x_new = x[*i];
      if(x_new < x_old){
        x_old = x_new;
        n_unique++;
      }
    }

    for(; i != order.end(); ++i){
      if(has_parent and parent[*i] == 0)
        continue;

      const double x_new = x[*i];
      if(x_new < x_first){
        start_knot = i;
        if(x_new < x_old)
          n_unique++;
        break;
      }
    }

    if(i == order.end())
      return { arma::vec(), n_unique };

    /* then the last knot */
    const arma::uword* ir = order.end() - 1;
    const arma::uword * const ir_end = order.begin() - 1;
    for(; has_parent and parent[*ir] == 0 and ir != ir_end; ir--);

    const double x_last  = x[*ir];
    x_old = x[*ir--];
    n_unique++;
    for(unsigned k = 1; k < endspan and ir != ir_end; --ir, ++k){
      if(has_parent and parent[*ir] == 0){
        k--;
        continue;
      }

      const double x_new = x[*ir];
      if(x_new > x_old){
        x_old = x_new;
        n_unique++;
      }
      if(&*ir == start_knot)
        break;
    }

    for(; ir != ir_end; --ir){
      if(has_parent and parent[*ir] == 0)
        continue;

      const double x_new = x[*ir];
      if(x_new > x_last or ir == start_knot){
        end_knot = ir;
        if(x_new > x_old)
          n_unique++;
        break;
      }
    }

    if(!end_knot or x[*end_knot] >= x[*start_knot]){
      /* we may have a single knot */
      n_unique--;
      if(n_unique > 2){
        arma::vec out(1);
        out.at(0) = x[*start_knot];
        return { std::move(out), n_unique };
      }
      return { arma::vec(), n_unique };
    }
  }

  /* find interior knots */
  arma::vec out(std::distance(start_knot, end_knot) + 1);
  out[0] = x[*start_knot];
  double x_old = out[0], *o = out.begin() + 1;
  std::size_t n_inner = 0;
  for(auto i = start_knot + 1; i != end_knot; i++){
    const double x_old_old = x_old;
    for(unsigned k = 1; i != end_knot and k < minspan; ++k, ++i){
      if(has_parent and parent[*i] == 0){
        k--;
        continue;
      }

      const double x_new = x[*i];
      if(x_new < x_old){
        n_unique++;
        x_old = x_new;
      }
    }
    for(; (i != end_knot and x[*i] >= x_old_old) or
          (i != end_knot and has_parent and parent[*i] == 0); ++i);
    if(i != end_knot){
      if(x[*i] < x_old)
        n_unique++;
      x_old = x[*i];
      *o++ = x_old;
      n_inner++;
      continue;
    } else
      break;
  }

  out[n_inner + 1] = x[*end_knot];

  return { out.subvec(0, n_inner + 1), n_unique };
}


/* function to call with minimum number of observations between each knot */
inline knot_res get_knots
  (const arma::vec &x, const unsigned endspan, const unsigned minspan,
   const arma::uvec &order, const arma::vec &parent){
  if(parent.n_elem < 1){
    if(endspan < 2L and minspan < 2L)
      return get_all_knots<false>(x, order, parent);

    return get_knots_w_span<false>(x, endspan, minspan, order, parent);
  } else {
    if(endspan < 2L and minspan < 2L)
      return get_all_knots<true>(x, order, parent);

    return get_knots_w_span<true>(x, endspan, minspan, order, parent);

  }
}


#endif
