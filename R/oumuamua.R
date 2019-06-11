#' @title Multivariate Adaptive Regression Splines Estimation
#' @description
#' Runs the Multivariate Adaptive Regression Splines (MARS)
#' algorithm potentially in parallel.
#'
#' @param formula a \code{\link{formula}} object with responses on the
#' left-hand-side of \code{~} and covariates on the right-hand-side of
#' \code{~}.
#' @param data \code{\link{environment}} or \code{\link{data.frame}} with
#' variables in \code{formula}.
#' @param offset a symbol or vector with a priori known component of the
#' linear predictor.
#' @param control \code{\link{list}} with arguments passed to
#' \code{\link{oumua.control}}.
#' @param x matrix with covariates.
#' @param y observed outcomes.
#' @param do_check logical for whether to perform checks of input arguments.
#'
#' @details
#' Runs the MARS algorithm as suggested in Friedman (1991). The README at
#' \url{https://github.com/boennecd/Oumuamua} contains simulations examples
#' which also illustrates how to use the functions in the package.
#'
#' @references
#' Friedman, Jerome H. \emph{Multivariate Adaptive Regression Splines}.
#' The Annals of Statistics 19.1 (1991): 1-67.
#'
#' @examples
#' library(Oumuamua)
#' data("mtcars")
#' f1 <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
#'             control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
#'                                     degree = 2, n_threads = 1))
#' f2 <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
#'             control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
#'                                     degree = 2, n_threads = 2))
#' stopifnot(all.equal(coef(f1), coef(f2)))
#'
#' @return
#' \code{oumua} returns an object of class oumua. The elements of the
#' returned object are
#' \item{X}{design matrix in final model.}
#' \item{n_vars}{number of basis functions in final model.}
#' \item{X_scales}{standard deviations of original covariates.}
#' \item{X_means}{means of original covariates.}
#' \item{drop_order}{order that basis functions are dropped in during the
#' backward pass.}
#' \item{backward_stats}{\eqn{R^2}s and generalized cross validation criteria
#' during backward pass.}
#' \item{root_childrens}{objects containing information about the estimated
#' tree.}
#' \item{Y}{observed outcomes used in estimation.}
#' \item{coefficients}{estimated coefficients.}
#' \item{all_nodes}{"unlisted" version of \code{root_childrens}.}
#' \item{call}{matched call.}
#' \item{terms}{\code{\link{terms.object}} from \code{\link{model.frame}}
#' call.}
#'
#' @importFrom stats model.response model.matrix model.offset model.frame
#' @export
oumua <- function(formula, data, offset, control = oumua.control()){
  # checks
  stopifnot(
    inherits(formula, "formula"), is.environment(data) || is.data.frame(data))

  # setup. First get model frame
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  # then the components
  tr <- terms(mf)
  attr(tr, "intercept") <- 0
  y <- model.response(mf, "numeric")
  x <- model.matrix(tr, mf)
  offset <- model.offset(mf)

  # fit model
  out <- oumua.fit(
    x = x, y = y, offset = offset, control = control)
  out[c("call", "terms")] <- list(cl, tr)
  class(out) <- "oumua"
  out
}

#' @export
print.oumua <- function(x, ...){
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")

  cat(sprintf("Including %d of %d terms after the backward pass\n\n",
              length(coef(x)) - 1, length(x$all_nodes)))
}

#' @title Summarizing MARS Fits
#' @description
#' Adds the \eqn{R^2} to the \code{oumua} object which can then be used in
#' \code{print.summary.oumua}.
#'
#' @param object object of class \code{oumua}.
#' @param ... un-used.
#'
#' @examples
#' library(Oumuamua)
#' data("mtcars")
#' fit <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
#'              control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
#'                                      degree = 2))
#' summary(fit)
#'
#' @importFrom stats coef
#' @export
summary.oumua <- function(object, ...){
  f <- drop(object$X %*% coef(object))
  y <- object$Y
  r <- y - f

  object$Rsq <- 1 - sum(r^2) / sum((y - mean(y))^2)
  class(object) <- "summary.oumua"
  object
}

#' @export
print.summary.oumua <- function(x, digits = 3, ...){
  print.oumua(x)

  m <- as.matrix(x$coefficients)
  colnames(m) <- "coefficients"
  print(m, digits = digits)

  cat(sprintf("\nR^2 is %5.3f\n", x$Rsq))

  invisible(x)
}

#' @rdname oumua
#' @export
oumua.fit <- function(x, y, offset = NULL, do_check = TRUE,
                      control = oumua.control()){
  N <- NROW(x)
  if(do_check)
    stopifnot(
      is.matrix(x), is.vector(y), length(y) == N,
      is.null(offset) || (is.numeric(offset) && length(offset) == N))
  if(!is.null(offset))
    y <- y - offset
  control <- do.call(oumua.control, control)

  endspan <- if(is.na(control$endspan))
    max(1L, as.integer(N / 100L)) else control$endspan
  minspan <- if(is.na(control$minspan))
    max(1L, as.integer(N / 100L)) else control$minspan

  # form tree
  fit <- omua_to_R(
    X = x, Y = y, lambda = control$lambda, endspan = endspan,
    minspan = minspan, degree = control$degree, nk = control$nk,
    penalty = control$penalty, trace = control$trace,
    thresh = control$thresh, n_threads = control$n_threads)
  fit$drop_order <- drop(fit$drop_order)
  fit$backward_stats <- lapply(fit$backward_stats, drop)
  fit$Y <- y
  fit$control <- control

  # find coefficients for selected model
  idx_obs <- seq_along(y)
  fX <- fit$X
  ey <- c(y, rep(0, NCOL(fX) - 1))
  Qo <- qr(fX, LAPACK = TRUE)
  fit$coefficients <- drop(backsolve(qr.R(Qo),  qr.qty(Qo, ey)[1:NCOL(fX)]))

  # account for pivoting
  idx <- Qo$pivot
  idx[idx] <- 1:length(idx)
  fit$coefficients <- fit$coefficients[idx]

  # format nodes in tree
  fit$root_childrens <- lapply(
    fit$root_childrens, format_node, x_means = fit$X_means,
    x_scales = fit$X_scales, x_names = colnames(x))

  # create list with all nodes
  fit$all_nodes <- fit$root_childrens
  func <- function(x){
    fit$all_nodes <<- c(fit$all_nodes, x$children)
    lapply(x$children, func)
  }
  lapply(fit$root_childrens, func)
  all_nodes_idx <- sapply(fit$all_nodes, "[[", "add_idx")
  fit$all_nodes <- fit$all_nodes[order(all_nodes_idx)]

  # set names
  fit$X <- fit$X[1:NROW(x), ]
  N <- length(fit$all_nodes)
  keep <- fit$drop_order[(N - fit$n_vars + 1):N] + 1
  names(fit$coefficients) <- colnames(fit$X) <- c(
    "(Intercept)",
    sapply(fit$all_nodes[keep], "[[", "description"))

  fit
}

# function to format nodes in tree
format_node <- function(node, x_means, x_scales, x_names, parent = NULL){
  class(node) <- "ouNode"
  if(is.nan(node$knot)){
    cov_idx <- node$cov_index + 1
    node$description <- x_names[cov_idx]
    if(!is.null(parent))
      node$description <- paste0(parent$description, ":", node$description)

    if(length(node$children) > 0)
      node$children <- lapply(
        node$children, format_node, x_means = x_means, x_scales = x_scales,
        x_names = x_names, parent = node)

    return(node)
  }

  # format description
  cov_idx <- node$cov_index + 1
  node$mean <- x_means[cov_idx]
  node$scale <- x_scales[cov_idx]

  digs <- 4
  off <- -node$sign * (node$knot * node$scale + node$mean)
  off <- signif(off, digs)
  sds <- signif(node$scale, digs)
  pm <- function(x)
    if(x < 0) as.character(x) else paste0("+", x)
  vterm <- paste0(if(node$sign < 0) "-" else "", x_names[cov_idx])
  node$description <- paste0("h((", vterm, pm(off), ")/", sds,")")

  if(!is.null(parent))
    node$description <- paste0(parent$description, ":", node$description)

  if(length(node$children) > 0)
    node$children <- lapply(
      node$children, format_node, x_means = x_means, x_scales = x_scales,
      x_names = x_names, parent = node)

  node
}

#' @export
print.ouNode <- function(x, ...)
  cat(x$description, "\n", sep = "")

#' @title Auxiliary for Controlling MARS Estimation
#' @description
#' Auxiliary function for \code{\link{oumua}} and \code{\link{oumua.fit}}.
#'
#' @param lambda L2 penalty mentioned in Friedman (1991, 32). Included both
#' for numerical stability and potentially better generalization.
#' @param endspan,minspan distance between first (last) observation and
#' respectively the first (last) knot and "internal" knots. See Friedman
#' (1991, 26-28).
#' @param degree maximum degree of basis terms. One yields an additive model.
#' @param nk maximum number of basis functions.
#' @param penalty penalty in generalized cross validation used in both the
#' forward and backward pass. See Friedman (1991, 19-22).
#' @param trace integer controlling amount of information that is printed to
#' the console during estimation. Zero yields no information.
#' @param thresh required improvement in \eqn{R^2} during forward pass to take
#' an additional iteration.
#' @param n_threads integer with number of threads to use.
#'
#' @references
#' Friedman, Jerome H. \emph{Multivariate Adaptive Regression Splines}.
#' The Annals of Statistics 19.1 (1991): 1-67.
#'
#' @examples
#' str(oumua.control())
#' str(oumua.control(n_threads = 6L))
#'
#' @export
oumua.control <- function(
  lambda = 1e-8, endspan = NA_integer_, minspan = NA_integer_, degree = 1L,
  nk = 20L, penalty = if(degree > 1) 3 else 2, trace = 0L, thresh = .001,
  n_threads = 1){
  stopifnot(
    is.numeric(lambda), length(lambda) == 1, lambda >= 0,
    is.integer(endspan), length(endspan) == 1, endspan > 0 || is.na(endspan),
    is.integer(minspan), length(minspan) == 1, minspan > 0 || is.na(minspan),
    is.integer(minspan), length(minspan) == 1, degree > 0,
    is.integer(nk), length(nk) == 1, nk > 1,
    is.numeric(penalty), length(penalty) == 1, penalty > 0,
    is.integer(trace), length(trace) == 1, trace > -1,
    is.numeric(thresh), length(thresh) == 1, thresh > 0,
    is.numeric(n_threads), length(n_threads) == 1, n_threads >= 1L)

  list(lambda = lambda, endspan = endspan, minspan = minspan, degree = degree,
       nk = nk, penalty = penalty, trace = trace, thresh = thresh,
       n_threads = n_threads)
}

#' @title Predict Method for MARS Fits
#' @description
#' Returns predicted values for new observations form an estimated MARS
#' model.
#'
#' @param object object of class \code{oumua}.
#' @param newdata \code{\link{data.frame}} with new observations.
#' @param ... un-used.
#'
#' @examples
#' library(Oumuamua)
#' data("mtcars")
#' fit <- oumua(mpg ~ cyl + disp + hp + drat + wt, mtcars,
#'              control = oumua.control(lambda = 1, endspan = 1L, minspan = 1L,
#'                                      degree = 2))
#' stopifnot(all.equal(drop(fit$X[1:10, ] %*% coef(fit)),
#'                     predict(fit, mtcars[1:10, ])))
#'
#' @importFrom stats terms
#' @export
predict.oumua <- function(object, newdata, ...){
  X <- model.matrix(object$terms, newdata)
  X <- get_design_map_from_R(
    object$root_childrens, X = X, X_scales = object$X_scales,
    X_means = object$X_means, drop_order = object$drop_order,
    n_vars = object$n_vars)

  drop(X %*% coef(object))
}
