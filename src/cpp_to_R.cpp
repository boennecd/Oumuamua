#include "Oumuamua.h"
#include "print.h"

// [[Rcpp::export]]
void omua_to_R
  (const arma::mat &X, const arma::vec &Y, const arma::vec &W,
   const double lambda, const unsigned endspan, const unsigned minspan,
   const unsigned degree, const unsigned nk, const double penalty,
   const unsigned trace, const double thresh){
  omua(X, Y, W, lambda, endspan, minspan, degree, nk, penalty, trace,
       thresh);
}
