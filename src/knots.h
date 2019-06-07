#ifndef KNOTS_H
#define KNOTS_H
#include "arma.h"

using std::size_t;

struct knot_res {
  arma::vec knots;
  size_t n_unique;

  operator const arma::vec&() const {
    return knots;
  }
};

inline void get_knots_check_input(const arma::vec &x){
#ifdef OUMU_DEBUG
{
  arma::uvec tmp = arma::find(arma::diff(x) > 0.);
  if(tmp.n_elem > 0L)
    throw std::invalid_argument("'get_all_knots': 'x' is not decreasing");
  if(x.n_elem < 3L)
    throw std::invalid_argument("'get_all_knots': too small 'x'");
}
#endif
}

/* takes sorted input and returns all interior unique values. Returns and
 * empty vector if input has two or fewer unqiue values. */
inline knot_res get_all_knots(const arma::vec &x){
  get_knots_check_input(x);

  arma::vec out(x.n_elem - 2L);
  size_t n_ele = 0L;
  double old = x[0L];
  double *o = out.begin();
  for(auto z = x.begin() + 1L; z != x.end(); old = *z, ++z)
    if(*z < old){
      *o++ = *z;
      n_ele++;
      if(o == out.end())
        break;
    }

  if(n_ele < 2L)
    return { arma::vec(), n_ele + 1L };

  return { out.subvec(0L, n_ele - 2L), n_ele + 1L };
}


/* function to call with minimum number of observations between each knot */
inline knot_res get_knots
  (const arma::vec &x, const unsigned endspan, const unsigned minspan){
  if(endspan < 2L and minspan < 2L)
    return get_all_knots(x);

  throw std::runtime_error("not implemented");
}


#endif
