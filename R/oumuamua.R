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

  # fit model and return
  out <- oumua.fit(
    x = x, y = y, offset = offset, weights = weights, control = control)
  out[c("call", "terms")] <- list(cl, tr)
  class(out) <- "oumua"
  out
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

  # endspan <- if(is.na(control$endspan))
  #   max(1L, as.integer(N / 10L)) else control$endspan
  # minspan <- if(is.na(control$minspan))
  #   max(1L, as.integer(N / 10L)) else control$minspan

  # TODO: change!
  endspan <- 1L
  minspan <- 1L

  fit <- omua_to_R(
    X = x, Y = y, W = weights, lambda = control$lambda, endspan = endspan,
    minspan = minspan, degree = control$degree, nk = control$nk,
    penalty = control$penalty, trace = control$trace,
    thresh = control$thresh)
}

#' @export
oumua.control <- function(
  lambda = 1e-8, endspan = NA_integer_, minspan = NA_integer_, degree = 1L,
  nk = 20L, penalty = if(degree > 1) 3 else 2, trace = 0L, thresh = .001){
  stopifnot(
    is.numeric(lambda), length(lambda) == 1, lambda > 0,
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
