#' @importFrom stats model.response model.matrix model.offset model.weights model.frame
#' @export
oumua <- function(formula, data, weights, offset, control = oumua.control()){
  # checks
  stopifnot(
    inherits(formula, "formula"), is.environment(data) || is.data.frame(data))

  # setup. First get model frame
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "weights", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  # then the components
  tr <- terms(mf)
  attr(tr, "intercept") <- 0
  y <- model.response(mf, "numeric")
  x <- model.matrix(tr, mf)
  weights <- model.weights(mf)
  offset <- model.offset(mf)

  # fit model
  out <- oumua.fit(
    x = x, y = y, offset = offset, weights = weights, control = control)
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

#' @export
summary.oumua <- function(object, ...){
  f <- drop(object$X %*% coef(object))
  w <- object$W
  r <- object$Y - f
  m <- sum(w * f) / sum(w)
  mss <- sum(w * (f - m)^2)
  rss <- sum(w * r^2)

  object$Rsq <- mss / (mss + rss)
  class(object) <- "summary.oumua"
  object
}

#' @export
print.summary.oumua <- function(x, digits = 3, ...){
  print.oumua(x)

  m <- as.matrix(x$coefficients)
  colnames(m) <- "coefficients"
  print(m, digits = digits)

  cat(sprintf("\nR^2 is %5.3f", x$Rsq))

  invisible(x)
}

#' @export
oumua.fit <- function(x, y, offset = NULL, weights = NULL, do_check = TRUE,
                      control = oumua.control()){
  N <- NROW(x)
  if(do_check)
    stopifnot(
      is.matrix(x), is.vector(y), length(y) == N, is.null(offset) || (
        is.numeric(offset) && length(offset) == N), is.null(weights) || (
          is.numeric(weights) && length(weights) == N), is.list(control))
  if(!is.null(offset))
    y <- y - offset
  if(is.null(weights))
    weights <- rep(1, N)
  control <- do.call(oumua.control, control)

  endspan <- if(is.na(control$endspan))
    max(1L, as.integer(N / 100L)) else control$endspan
  minspan <- if(is.na(control$minspan))
    max(1L, as.integer(N / 100L)) else control$minspan

  # form tree
  fit <- omua_to_R(
    X = x, Y = y, W = weights, lambda = control$lambda, endspan = endspan,
    minspan = minspan, degree = control$degree, nk = control$nk,
    penalty = control$penalty, trace = control$trace,
    thresh = control$thresh)
  fit$drop_order <- drop(fit$drop_order)
  fit$backward_stats <- lapply(fit$backward_stats, drop)
  fit$Y <- y
  fit$W <- weights
  fit$control <- control

  # find coefficients for selected model
  sqrt_w <- sqrt(weights)
  idx_obs <- seq_along(y)
  wX <- fit$X
  wX[idx_obs, ] <- wX[idx_obs, ] * sqrt_w
  wy <- c(y * sqrt_w, rep(0, NCOL(wX) - 1))
  Qo <- qr(wX, LAPACK = TRUE)
  fit$coefficients <- drop(backsolve(qr.R(Qo),  qr.qty(Qo, wy)[1:NCOL(wX)]))

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

#' @export
oumua.control <- function(
  lambda = 1e-8, endspan = NA_integer_, minspan = NA_integer_, degree = 1L,
  nk = 20L, penalty = if(degree > 1) 3 else 2, trace = 0L, thresh = .001){
  stopifnot(
    is.numeric(lambda), length(lambda) == 1, lambda >= 0,
    is.integer(endspan), length(endspan) == 1, endspan > 0 || is.na(endspan),
    is.integer(minspan), length(minspan) == 1, minspan > 0 || is.na(minspan),
    is.integer(minspan), length(minspan) == 1, degree > 0,
    is.integer(nk), length(nk) == 1, nk > 1,
    is.numeric(penalty), length(penalty) == 1, penalty > 0,
    is.integer(trace), length(trace) == 1, trace > -1,
    is.numeric(thresh), length(thresh) == 1, thresh > 0)

  list(lambda = lambda, endspan = endspan, minspan = minspan, degree = degree,
       nk = nk, penalty = penalty, trace = trace, thresh = thresh)
}

#' @export
predict.oumua <- function(x, newdata){
  X <- model.matrix(x$terms, newdata)
  X <- get_design_map_from_R(
    x$root_childrens, X = X, X_scales = x$X_scales, X_means = x$X_means,
    drop_order = x$drop_order, n_vars = x$n_vars)

  drop(X %*% coef(x))
}
