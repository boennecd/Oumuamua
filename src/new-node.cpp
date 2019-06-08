#include "new-node.h"
#include "blas-lapack.h"

using std::to_string;

static constexpr char C_N = 'N';
static constexpr double D_ONE = 1;
static constexpr int I_ONE = 1;

inline arma::span get_sold(const bool fresh, const unsigned p){
  return fresh ? arma::span() : arma::span(0L, p - 1L);
}

inline void check_new_node_input
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const double lambda,
   const unsigned N, const std::string &msg_prefix,
   const arma::vec *knots, const bool check_order){
#ifdef OUMU_DEBUG
  const arma::uword n = x.n_elem, p = B.n_rows;

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
  if(B.n_cols != n or y.n_elem != n or parent.n_elem != n)
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
  const arma::uword n = x.n_elem, p = B.n_rows;
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
    V.rows(sold) = B * x_parent;

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
   const double lambda, const unsigned N, const bool one_hinge){
  const arma::uword n = x.n_elem, p = B.n_rows,
    idx_last_term = p + !one_hinge;
  const bool fresh = p < 1L;
  const double dN = N;

  check_new_node_input(
    old_problem, x, y, parent, B, lambda, N, "'get_new_node': ",
    &knots, true);

  arma::vec x_cen;
  normal_equation problem_to_update;
  if(!one_hinge){
    /* Handle the first part for the term with the idenity function */
    const auto lin_term_obj = add_linear_term
      (old_problem, x, y, parent, B, lambda, N);
    x_cen = lin_term_obj.x_cen;
    problem_to_update = lin_term_obj.new_eq;

  } else
    problem_to_update = old_problem;
  problem_to_update.resize(problem_to_update.n_elem() + 1);

  /* prep for going through knot */
  const arma::span sold = get_sold(fresh, p);
  arma::mat V(p + 1L + !one_hinge, 1L, arma::fill::zeros);
  arma::vec k(1L, arma::fill::zeros);

  /* handle penalty term for hinge function */
  V(idx_last_term, 0L) = lambda;

  /* loop over knots and find the one with the lowest squared error */
  new_node_res out;
  double knot_old = knots.at(0L);
  /* we have to keep track of new active observations and those that were
   * active from the previous iteration */
  arma::uword active_end = 0L, new_active_end = 0L;
  const arma::vec parent_sq = arma::square(parent);

  /* quantites in equation (52) in Friedman (1991) */
  double grad_term_old = 0., V_x_h = 0., parent_sq_h = 0., sum_parent_sq = 0.,
    su = 0., sum_parent = 0.;
  arma::mat V_old_h(p, 1, arma::fill::zeros);

  /* TODO: replace with BLAS calls? */
  const int m_B = B.n_rows;
  for(const double *knot = knots.begin(); knot != knots.end();
      knot_old = *knot, ++knot, active_end = new_active_end){
    /* find new_end */
    {
      const double *xi = x.begin() + active_end;
      for(new_active_end = active_end; new_active_end < n and *xi > *knot;
          ++new_active_end, ++xi);
    }

    /* we need to compute two squares of sums */
    double sl = 0.;
    const arma::span new_active(active_end, new_active_end - 1L);

    /* make update for *new* active observations */
    {
      const arma::vec par_x_less_knot =
        parent(new_active) % (x(new_active) - *knot);
      k.at(0L) += arma::dot(y(new_active), par_x_less_knot);

      if(!fresh){
        const int n = new_active.b - new_active.a + 1;
        F77_CALL(dgemv)(
          &C_N, &m_B, &n, &D_ONE, B.colptr(new_active.a), &m_B,
          par_x_less_knot.memptr(), &I_ONE, &D_ONE, V.memptr(), &I_ONE);
      }

      if(!one_hinge)
        V(p, 0L) += arma::dot(x_cen(new_active), par_x_less_knot);

      V(idx_last_term, 0L) +=
        arma::dot(par_x_less_knot, par_x_less_knot);

      sl += arma::sum(par_x_less_knot);
    }

    /* make update on active observations */
    const double knot_diff = knot_old - *knot;
    k.at(0L) += knot_diff * grad_term_old;
    if(!fresh)
      V.rows(sold) += knot_diff * V_old_h;
    if(!one_hinge)
      V(p, 0L) += knot_diff * V_x_h;
    sl += su + sum_parent * knot_diff;
    V(idx_last_term, 0L) +=
      knot_diff * (2. * parent_sq_h - sum_parent_sq * (knot_old + *knot)) +
      (su * su - sl * sl) / dN;

    /* update intermediaries */
    grad_term_old += arma::dot(y(new_active), parent(new_active));
    if(!fresh){
      const int n = new_active.b - new_active.a + 1;
      F77_CALL(dgemv)(
        &C_N, &m_B, &n, &D_ONE, B.colptr(new_active.a),
        &m_B, parent.memptr() + new_active.a, &I_ONE,
        &D_ONE, V_old_h.memptr(), &I_ONE);
    }
    if(!one_hinge)
      V_x_h += arma::dot(x_cen(new_active), parent(new_active));
    parent_sq_h += arma::dot(parent_sq(new_active), x(new_active));
    for(unsigned i = new_active.a; i <= new_active.b; ++i){
      sum_parent    += parent   .at(i);
      sum_parent_sq += parent_sq.at(i);
    }
    su = sl;

    /* update solution */
    problem_to_update.update_sub(V, k);
    const double se = get_min_se_less_var(problem_to_update, lambda);
    if(se < out.min_se_less_var){
      out.min_se_less_var = se;
      out.knot = *knot;
    }
  }

  return out;
}
