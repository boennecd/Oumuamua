
Oumuamua
========

Additive function
-----------------

``` r
# generates correlated uniform variables 
get_covs <- function(N, p, rho = .5){
  X <- matrix(rnorm(N * p), nc = p)
  Sig <- diag(p)
  Sig[lower.tri(Sig)] <- Sig[upper.tri(Sig)] <- rho
  X <- X %*% chol(Sig)
  apply(X, 2, pnorm)
}

# simulates data 
additiv_sim <- function(N, p){
  p <- max(5L, p)
  x <- get_covs(N = N, p = p)
  y <- .1 * exp(4 * x[, 1]) + 4 / (1 + exp(-20 * (x[, 2] - 1/2))) + 
    3 * x[, 3] + 2 * x[, 4] + x[, 5] + rnorm(N)
  data.frame(y = y, x)
}
```

``` r
library(earth)
library(Oumuamua)
```

``` r
# parameters in simulation
test_size <- 10000
N <- c(100, 200, 500)
p <- 10
```

``` r
# returns minspan and endspan arguments. Similar to suggestion in Friedman 
# (1991) though without ajdusting N for number of striclty positive elements 
# in the basis function.
get_spans <- function(N, p, alpha = .05){
  Np <- N * p
  minspan <- as.integer(ceiling(-log2(-1/Np * log1p(-alpha)) / 2.5))
  endspan <- as.integer(ceiling(3 -log2(alpha / p)))
  c(minspan = minspan, endspan = endspan)
}

# functions to fit models
earth_call <- function(sims){
  spans <- get_spans(N = nrow(sims), p = p)
  earth(y ~ ., data = sims, minspan = spans["minspan"], 
        endspan = spans["endspan"], degree = 1, penalty = 2)
}
oumua_call <- function(sims, n_threads = 5L){
  spans <- get_spans(N = nrow(sims), p = p)
  oumua(y ~ ., data = sims, control = oumua.control(
    minspan = spans["minspan"], endspan = spans["endspan"], degree = 1L, 
    penalty = 2, lambda = 1, n_threads = n_threads))
}

# run simulations
set.seed(3779892)
res <- lapply(N, function(N_i){
  # data used for validation
  test_dat <- additiv_sim(test_size, p)
  
  replicate(1000, {
    # simulate
    sims <- additiv_sim(N_i, p)
    
    # fit models
    efit <- earth_call(sims)
    ofit <- oumua_call(sims)
    
    # compute MSE and return
    mse <- function(fit){
      yhat <- predict(fit, newdata = test_dat)
      mean((test_dat$y - yhat)^2)
    }
    
    c(earth = mse(efit), oumua = mse(ofit))
  })
})
```

``` r
# stats for mean square error
names(res) <- N
lapply(res, function(x) apply(x, 1, function(z) 
  c(mean = mean(z), `standard error` = sd(z) / sqrt(length(z)))))
#> $`100`
#>                   earth    oumua
#> mean           1.441382 1.441227
#> standard error 0.005773 0.004895
#> 
#> $`200`
#>                   earth    oumua
#> mean           1.207552 1.193977
#> standard error 0.002465 0.001961
#> 
#> $`500`
#>                    earth     oumua
#> mean           1.0826603 1.0777904
#> standard error 0.0008677 0.0007799
```

``` r
library(microbenchmark)
```

``` r
set.seed(17039344)
addi_runtimes <- local({
  run_dat <- additiv_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), 
    `oumua (1 thread) ` = oumua_call(run_dat, n_threads = 1L),
    `oumua (5 threads)` = oumua_call(run_dat, n_threads = 5L),
    times = 100)
})
```

``` r
addi_runtimes
#> Unit: milliseconds
#>               expr   min    lq  mean median    uq    max neval
#>              earth 60.07 61.79 65.80  62.99 64.04 102.61   100
#>  oumua (1 thread)  57.04 58.63 59.78  59.26 60.18  75.79   100
#>  oumua (5 threads) 27.26 28.02 29.56  28.78 30.12  36.76   100
```

Interaction Example
-------------------

``` r
interact_sim <- function(N, p){
  p <- max(5L, p)
  x <- get_covs(N = N, p = p)
  y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 1/2)^2 + 
    10 * x[, 4] + 5 * x[, 5] + rnorm(N)
  data.frame(y = y, x)
}
```

``` r
# functions to fit models
earth_call <- function(sims){
  spans <- get_spans(N = nrow(sims), p = p)
  earth(y ~ ., data = sims, minspan = spans["minspan"], 
        endspan = spans["endspan"], degree = 3, penalty = 3, nk = 30, 
        fast.k = 0)
}
oumua_call <- function(sims, n_threads = 5L){
  spans <- get_spans(N = nrow(sims), p = p)
  oumua(y ~ ., data = sims, control = oumua.control(
    minspan = spans["minspan"], endspan = spans["endspan"], degree = 3L, 
    penalty = 3, nk = 30L, lambda = 1, n_threads = n_threads))
}

# run simulations
set.seed(3779892)
res <- lapply(N, function(N_i){
  # data used for validation
  test_dat <- interact_sim(test_size, p)
  
  replicate(1000, {
    # simulate
    sims <- interact_sim(N_i, p)
    
    # fit models
    efit <- earth_call(sims)
    ofit <- oumua_call(sims)
    
    # compute MSE and return
    mse <- function(fit){
      yhat <- predict(fit, newdata = test_dat)
      mean((test_dat$y - yhat)^2)
    }
    
    c(earth = mse(efit), oumua = mse(ofit))
  })
})
```

``` r
# stats for mean square error
names(res) <- N
lapply(res, function(x) apply(x, 1, function(z) 
  c(mean = mean(z), `standard error` = sd(z) / sqrt(length(z)))))
#> $`100`
#>                  earth   oumua
#> mean           2.43652 2.76427
#> standard error 0.03568 0.03204
#> 
#> $`200`
#>                 earth    oumua
#> mean           1.6460 1.485734
#> standard error 0.0119 0.005562
#> 
#> $`500`
#>                   earth    oumua
#> mean           1.252763 1.220402
#> standard error 0.002891 0.002427
```

``` r
set.seed(17039344)
inter_runtimes <- local({
  run_dat <- interact_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), 
    `oumua (1 thread) ` = oumua_call(run_dat, n_threads = 1L),
    `oumua (5 threads)` = oumua_call(run_dat, n_threads = 5L),
    times = 10)
})
```

``` r
inter_runtimes
#> Unit: milliseconds
#>               expr   min    lq  mean median    uq   max neval
#>              earth 550.6 555.5 569.2  566.0 575.9 603.9    10
#>  oumua (1 thread)  806.6 824.1 833.1  831.0 836.7 865.0    10
#>  oumua (5 threads) 193.2 197.1 215.5  209.8 223.3 259.2    10
```

Interaction Example with Factor
-------------------------------

``` r
factor_sim <- function(N, p){
  p <- max(5L, p)
  x <- get_covs(N = N, p = p)
  
  n_grp <- 5L
  fac <- gl(n_grp, N / n_grp)
  grp_effect <- seq(-3, 3, length.out = n_grp) 
    
  y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 1/2)^2 + 
    10 * x[, 4] + 5 * x[, 5] + grp_effect[as.integer(fac)] + rnorm(N)
  data.frame(y = y, x, fac = fac)
}
```

``` r
# functions to fit models
earth_call <- function(sims){
  spans <- get_spans(N = nrow(sims), p = p)
  earth(y ~ ., data = sims, minspan = spans["minspan"], 
        endspan = spans["endspan"], degree = 3, penalty = 3, nk = 30, 
        fast.k = 0, fast.beta = 0)
}
oumua_call <- function(sims, lambda = 1, n_threads = 5L){
  spans <- get_spans(N = nrow(sims), p = p)
  oumua(y ~ ., data = sims, control = oumua.control(
    minspan = spans["minspan"], endspan = spans["endspan"], degree = 3L, 
    penalty = 3, lambda = lambda, nk = 30L, n_threads = n_threads))
}

# run simulations
set.seed(3779892)
res <- lapply(N, function(N_i){
  # data used for validation
  test_dat <- factor_sim(test_size, p)
  
  replicate(1000, {
    # simulate
    sims <- factor_sim(N_i, p)
    
    # fit models
    efit <- earth_call(sims)
    ofit <- oumua_call(sims)
    
    # compute MSE and return
    mse <- function(fit){
      yhat <- predict(fit, newdata = test_dat)
      mean((test_dat$y - yhat)^2)
    }
    
    c(earth = mse(efit), oumua = mse(ofit))
  })
})
```

``` r
# stats for mean square error
names(res) <- N
lapply(res, function(x) apply(x, 1, function(z) 
  c(mean = mean(z), `standard error` = sd(z) / sqrt(length(z)))))
#> $`100`
#>                  earth   oumua
#> mean           3.23292 4.86330
#> standard error 0.04681 0.06104
#> 
#> $`200`
#>                  earth    oumua
#> mean           1.75911 1.594299
#> standard error 0.01225 0.008598
#> 
#> $`500`
#>                   earth    oumua
#> mean           1.389467 1.249641
#> standard error 0.004928 0.003456
```

``` r
set.seed(17039344)
factor_runtimes <- local({
  run_dat <- factor_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), 
    `oumua (1 thread) ` = oumua_call(run_dat, n_threads = 1L),
    `oumua (5 threads)` = oumua_call(run_dat, n_threads = 5L),
    times = 10)
})
```

``` r
factor_runtimes
#> Unit: milliseconds
#>               expr    min     lq   mean median     uq    max neval
#>              earth  963.1  992.2 1012.3 1002.6 1025.0 1072.4    10
#>  oumua (1 thread)  1676.7 1716.6 1815.5 1791.9 1938.1 1990.6    10
#>  oumua (5 threads)  469.2  540.5  565.5  571.8  587.5  641.7    10
```

``` r
set.seed(17039344)
factor_runtimes <- local({
  run_dat <- factor_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), 
    `oumua (1 thread) ` = oumua_call(run_dat, n_threads = 1L, lambda = 0),
    `oumua (5 threads)` = oumua_call(run_dat, n_threads = 5L, lambda = 0),
    times = 10)
})
```

``` r
factor_runtimes
#> Unit: milliseconds
#>               expr    min     lq   mean median     uq    max neval
#>              earth  927.5  981.9 1004.9  990.7 1011.3 1108.2    10
#>  oumua (1 thread)  1080.6 1137.0 1185.4 1194.1 1204.8 1282.4    10
#>  oumua (5 threads)  283.1  311.4  354.6  369.3  390.7  410.8    10
```
