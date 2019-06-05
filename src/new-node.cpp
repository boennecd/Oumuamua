#include "new-node.h"

using std::to_string;

inline arma::span get_sold(const bool fresh, const unsigned p){
  return fresh ? arma::span() : arma::span(0L, p - 1L);
}

inline void check_new_node_input
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const double lambda,
   const unsigned N, const std::string &msg_prefix,
   const arma::vec *knots, const bool check_order){
#ifdef OUMU_DEBUG
  const arma::uword n = x.n_elem, p = B.n_cols;

  auto invalid_arg = [&](const std::string &msg){
    throw std::invalid_argument(msg_prefix + msg);
  };
  if(N < n)
    invalid_arg("too small 'N' (" + to_string(N) + ", " + to_string(n) + ")");
  if(knots){
    if(knots->n_elem > n - 2L)
      invalid_arg("too many knots (" + to_string(knots->n_elem) + ", " +
        to_string(n) + ")");
    if(knots->operator[](0L) >= x[0L])
      invalid_arg("first knot is not an interior knot");
    if(knots->tail(1L)(0L) <= x.tail(1L)(0L))
      invalid_arg("last knot is not an interior knot");
  }
  if(old_problem.n_elem() != p)
    invalid_arg("invalid 'old_problem' or 'B' (" +
      to_string(old_problem.n_elem()) + ", " + to_string(p) + ")");
  if(B.n_rows != n or y.n_elem != n or parent.n_elem != n)
    invalid_arg("invalid 'B', 'parent', or 'y'");
  if(check_order){
      arma::vec tmp = arma::diff(x);
      arma::uvec utmp = arma::find(tmp > 0.);
      if(utmp.n_elem > 0L)
        invalid_arg("'x' is not decreasing");

      if(knots){
        tmp = arma::diff(*knots);
        utmp = arma::find(tmp >= 0.);
        if(utmp.n_elem > 0L)
          invalid_arg("'knots' is not decreasing");
      }
  }
#endif
}

add_linear_term_res add_linear_term
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const double lambda,
   const unsigned N){
  const arma::uword n = x.n_elem, p = B.n_cols;
  const bool fresh = p < 1L;
  const double dN = N;

  check_new_node_input(
    old_problem, x, y, parent, B, lambda, N, "'add_linear_term': ",
    nullptr, false);

  const arma::span sold = get_sold(fresh, p);
  arma::mat V(p + 1L, 1L, arma::fill::zeros);
  arma::vec k(1L, arma::fill::zeros);
  const arma::vec x_parent = x % parent;
  const double x_parent_mean = arma::sum(x_parent) / dN;
  const arma::vec x_cen = x_parent - x_parent_mean;
  if(!fresh)
    V.rows(sold) = B.t() * x_parent;

  for(auto z : x_cen)
    V(p, 0L) += z * z;
  V(p, 0L) += lambda + x_parent_mean * x_parent_mean * (dN - n);
  k.at(0) = arma::dot(x_parent, y);

  normal_equation out = old_problem;
  out.update(V, k);
  return { std::move(out), std::move(x_cen) };
}

new_node_res get_new_node
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const arma::vec &knots,
   const double lambda, const unsigned N){
  const arma::uword n = x.n_elem, p = B.n_cols, p1 = p + 1;
  const bool fresh = p < 1L;
  const double dN = N;

  check_new_node_input(
    old_problem, x, y, parent, B, lambda, N, "'get_new_node': ",
    &knots, true);

  /* Handle the first part for the term with the idenity function */
  const auto lin_term_obj = add_linear_term
    (old_problem, x, y, parent, B, lambda, N);
  const arma::vec &x_cen = lin_term_obj.x_cen;
  const normal_equation &problem_w_lin_term = lin_term_obj.new_eq;

  /* prep for going through knot */
  const arma::span sold = get_sold(fresh, p);
  arma::mat V(p + 2L, 1L, arma::fill::zeros);
  arma::vec k(1L, arma::fill::zeros);

  /* handle penalty term for hinge function */
  V(p1, 0L) = lambda;

  /* loop over knots and find the one with the lowest squared error */
  new_node_res out;
  double knot_old = knots.at(0L);
  /* we have to keep track of new active observations and those that were
   * active from the previous iteration */
  arma::uword active_end = 0L, new_active_end = 0L;
  const arma::vec parent_sq = arma::square(parent);
  /* TODO: replace with BLAS calls? */
  for(const double *knot = knots.begin(); knot != knots.end();
      knot_old = *knot, ++knot, active_end = new_active_end){
    /* find new_end */
    {
      const double *xi = x.begin() + active_end;
      for(new_active_end = active_end; new_active_end < n and *xi > *knot;
          ++new_active_end, ++xi);
    }

    /* we need to compute two squares of sums */
    double su = 0., sl = 0.;

    /* make update for *new* active observations */
    {
      const arma::span new_active(active_end, new_active_end - 1L);
      const arma::vec par_x_less_knot =
        parent(new_active) % (x(new_active) - *knot);
      k.at(0L) += arma::dot(y(new_active), par_x_less_knot);

      if(!fresh)
        V.rows(sold) +=
          B.rows(new_active).t() * par_x_less_knot;

      V(p, 0L) += arma::dot(x_cen(new_active), par_x_less_knot);

      V(p1, 0L) +=
        arma::dot(par_x_less_knot, par_x_less_knot);

      sl += arma::sum(par_x_less_knot);
    }

    /* make update on active observations */
    if(active_end > 0){
      const double knot_diff = knot_old - *knot;
      const arma::span active(0L, active_end - 1L);
      k.at(0L) += knot_diff * arma::dot(y(active), parent(active));

      if(!fresh)
        V.rows(sold) +=
          knot_diff * (B.rows(active).t() *  parent(active));

      V(p, 0L) += knot_diff * arma::dot(x_cen(active), parent(active));
      V(p1, 0L) +=
        knot_diff *
        arma::dot(parent_sq(active), 2 * x(active) - knot_old - *knot);

      for(unsigned i = 0; i < active_end; ++i){
        su += parent.at(i) * (x.at(i) - knot_old);
        sl += parent.at(i) * (x.at(i) - *knot);
      }
    }

    /* handle two squared sums in lower right entry of Gramian matrix */
    V(p1, 0L) += (su * su - sl * sl) / dN;

    /* update solution */
    normal_equation new_problem = problem_w_lin_term;
    new_problem.update(V, k);
    const double se = get_min_se_less_var(new_problem, lambda);
    if(se < out.min_se_less_var){
      out.min_se_less_var = se;
      out.knot = *knot;
    }
  }

  return out;
}
