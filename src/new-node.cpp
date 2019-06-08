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
   const arma::vec *knots, const bool check_order, const arma::uvec &indices){
#ifdef OUMU_DEBUG
  const arma::uword n = x.n_elem, p = B.n_rows;
  const arma::vec x_sort = x(indices);

  auto invalid_arg = [&](const std::string &msg){
    throw std::invalid_argument(msg_prefix + msg);
  };
  if(N < n)
    invalid_arg("too small 'N' (" + to_string(N) + ", " + to_string(n) + ")");
  if(knots){
    if(knots->n_elem > n - 2L)
      invalid_arg("too many knots (" + to_string(knots->n_elem) + ", " +
        to_string(n) + ")");
    if(knots->operator[](0L) >= x_sort[0L])
      invalid_arg("first knot is not an interior knot");
    if(knots->tail(1L)(0L) <= x_sort.tail(1L)(0L))
      invalid_arg("last knot is not an interior knot");
  }
  if(old_problem.n_elem() != p)
    invalid_arg("invalid 'old_problem' or 'B' (" +
      to_string(old_problem.n_elem()) + ", " + to_string(p) + ")");
  if(B.n_cols != n or y.n_elem != n or parent.n_elem != n)
    invalid_arg("invalid 'B', 'parent', or 'y'");
  if(check_order){
      arma::vec tmp = arma::diff(x_sort);
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
   const unsigned N, const arma::uvec &indices){
  const arma::uword n = indices.n_elem, p = B.n_rows;
  const bool fresh = p < 1L;
  const double dN = N;

  check_new_node_input(
    old_problem, x, y, parent, B, lambda, N, "'add_linear_term': ",
    nullptr, false, indices);

  const arma::span sold = get_sold(fresh, p);
  arma::mat V(p + 1L, 1L, arma::fill::zeros);
  arma::vec k(1L, arma::fill::zeros);

  double x_parent_mean;
  if(x.n_elem == indices.n_elem){
    /* sepecial case when all observations are included */
    const arma::vec x_parent = x % parent;
    x_parent_mean = arma::sum(x_parent) / dN;
    const arma::vec x_cen = x_parent - x_parent_mean;
    if(!fresh)
      V.rows(sold) = B * x_parent;

    for(auto z : x_cen)
      V(p, 0L) += z * z;
    V(p, 0L) += lambda + x_parent_mean * x_parent_mean * (dN - n);
    k.at(0) = arma::dot(x_parent, y);

  } else {
    x_parent_mean = 0;
    for(auto idx : indices){
      const double x_parent = x[idx] * parent[idx];
      x_parent_mean += x_parent;
      if(!fresh)
        V.rows(sold) += B.col(idx) * x_parent;
      k.at(0) += x_parent * y[idx];

    }
    x_parent_mean /= dN;

    /* TODO: avoid this loop */
    for(auto idx : indices){
      const double x_cen = x[idx] * parent[idx] - x_parent_mean;
      V(p, 0L) += x_cen * x_cen;
    }
    V(p, 0L) += lambda + x_parent_mean * x_parent_mean * (dN - n);
  }

  normal_equation out = old_problem;
  out.update(V, k);
  return { std::move(out), x_parent_mean };
}

new_node_res get_new_node
  (const normal_equation &old_problem, const arma::vec &x, const arma::vec &y,
   const arma::vec &parent, const arma::mat &B, const arma::vec &knots,
   const double lambda, const unsigned N, const bool one_hinge,
   const arma::uvec &indices){
  const arma::uword p = B.n_rows,
    idx_last_term = p + !one_hinge;
  const bool fresh = p < 1L;
  const double dN = N;

  check_new_node_input(
    old_problem, x, y, parent, B, lambda, N, "'get_new_node': ",
    &knots, true, indices);

  double x_parent_mean = 0;
  normal_equation problem_to_update;
  if(!one_hinge){
    /* Handle the first part for the term with the idenity function */
    const auto lin_term_obj = add_linear_term
      (old_problem, x, y, parent, B, lambda, N, indices);
    x_parent_mean = lin_term_obj.x_parent_mean;
    problem_to_update = lin_term_obj.new_eq;

  } else
    problem_to_update = old_problem;
  problem_to_update.resize(problem_to_update.n_elem() + 1);
  /* TODO: delete */
  const arma::vec x_cen = x % parent - x_parent_mean;

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
  const arma::uword *active_end = indices.begin(),
                *new_active_end = indices.begin();

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
      for(new_active_end = active_end;
          x[*new_active_end] > *knot and new_active_end != indices.end();
          ++new_active_end);
    }

    /* we need to compute two squares of sums */
    double sl = 0.;
    for(const arma::uword *i = active_end; i != new_active_end; ++i){
      const double x_cen = x.at(*i) * parent.at(*i) - x_parent_mean;
      const double parent_i = parent.at(*i),
                parent_i_sq = parent_i * parent_i,
                        x_i = x.at(*i),
                        y_i = y.at(*i),
            par_x_less_knot = parent_i * (x_i - *knot);

      /* make update for *new* active observations */
      {
        k.at(0) += y_i * par_x_less_knot;

        if(!fresh)
          F77_CALL(daxpy)(
              &m_B, &par_x_less_knot, B.colptr(*i), &I_ONE,
              V.memptr(), &I_ONE);

        if(!one_hinge)
          V(p, 0L) += x_cen * par_x_less_knot;

        V(idx_last_term, 0L) += par_x_less_knot * par_x_less_knot;

        sl += par_x_less_knot;
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
      grad_term_old += y_i * parent_i;
      if(!fresh)
        F77_CALL(daxpy)(
          &m_B, parent.memptr() + *i, B.colptr(*i), &I_ONE, V_old_h.memptr(),
          &I_ONE);

      if(!one_hinge)
        V_x_h += x_cen * parent_i;
      parent_sq_h   += parent_i_sq * x_i;
      sum_parent    += parent_i;
      sum_parent_sq += parent_i_sq;
      su = sl;
    }

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
