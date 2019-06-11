
Oumuamua
========

[![Build Status on Travis](https://travis-ci.org/boennecd/Oumuamua.svg?branch=master,osx)](https://travis-ci.org/boennecd/Oumuamua)

The `Oumuamua` contains a parallel implementation of the Multivariate Adaptive Regression Splines algorithm suggested in Friedman (1991). The package can be installed from Github by calling

``` r
devtools::install_github("boennecd/Oumuamua")
```

There is yet no backronym for the package. The rest of this README contains simulation examples where this package is compared with the `earth` package. The simulation examples are from Friedman (1991) and are shown to illustrate how to use the package. A discussion about implementation differences between the `earth` package and this package is given at the end. Some comments about future tasks are also listed at the end.

Additive function
-----------------

We start of with a model with the additive model shown in Friedman (1991, 35). The model is

![y = 0.1 \\exp 4 x\_1 + \\frac 4{1 + \\exp-20(x\_2 - 1/2)} + 3x\_3 + 2x\_4 + x\_5 + 
   \\epsilon](https://latex.codecogs.com/svg.latex?y%20%3D%200.1%20%5Cexp%204%20x_1%20%2B%20%5Cfrac%204%7B1%20%2B%20%5Cexp-20%28x_2%20-%201%2F2%29%7D%20%2B%203x_3%20%2B%202x_4%20%2B%20x_5%20%2B%20%0A%20%20%20%5Cepsilon "y = 0.1 \exp 4 x_1 + \frac 4{1 + \exp-20(x_2 - 1/2)} + 3x_3 + 2x_4 + x_5 + 
   \epsilon")

where ![\\epsilon](https://latex.codecogs.com/svg.latex?%5Cepsilon "\epsilon") follows a standard normal distribution and the ![x\_i](https://latex.codecogs.com/svg.latex?x_i "x_i")s are uniformly distributed on ![(0,1)](https://latex.codecogs.com/svg.latex?%280%2C1%29 "(0,1)"). Moreover, we let the ![x\_i](https://latex.codecogs.com/svg.latex?x_i "x_i")s be correlated and introduce a fixed number of noisy correlated variables.

We start by defining two function to simulate the covariates and outcomes.

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

Then we run the simulations.

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
# (1991) though without adjusting N for number of striclty positive elements 
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

We make 1000 simulation above for the different sample sizes in the vector `N`. We use `p` covariates (10) of which only 5 are associated with the outcome. We only allow for an additive model by setting `degree = 1`. The `penalty` argument is parameter in the generalized cross validation criteria mentioned in Friedman (1991, 19–22). The `lambda` parameter is the L2 penalty mentioned in Friedman (1991, 32). The mean squared errors for each sample size is shown below.

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

A comparison of computation times with both 1 and 5 threads is given below.

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
#>               expr   min    lq  mean median    uq   max neval
#>              earth 59.88 62.09 64.96  63.13 64.71 108.4   100
#>  oumua (1 thread)  56.40 58.70 60.12  59.56 61.38  67.6   100
#>  oumua (5 threads) 26.93 27.57 29.26  28.10 29.71  59.4   100
```

Interaction Example
-------------------

Next, we consider the non-additive model in Friedman (1991, 37). The true model is

![y = \\sin\\pi x\_1x\_2 + 20(x\_3 - 1/2)^2 + 10x\_4 + 5x\_5 + \\epsilon](https://latex.codecogs.com/svg.latex?y%20%3D%20%5Csin%5Cpi%20x_1x_2%20%2B%2020%28x_3%20-%201%2F2%29%5E2%20%2B%2010x_4%20%2B%205x_5%20%2B%20%5Cepsilon "y = \sin\pi x_1x_2 + 20(x_3 - 1/2)^2 + 10x_4 + 5x_5 + \epsilon")

We define a function to simulate the covariates and outcomes.

``` r
interact_sim <- function(N, p){
  p <- max(5L, p)
  x <- get_covs(N = N, p = p)
  y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 1/2)^2 + 
    10 * x[, 4] + 5 * x[, 5] + rnorm(N)
  data.frame(y = y, x)
}
```

Then we perform the simulation.

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

We have increased `degree` to allow for interactions. We also increase `penalty` as suggested in Friedman (1991) (though some further tuning might be needed). We set `fast.k = 0` in the `earth` call as this feature is not yet implemented in this package. The mean squared error estimate is shown below.

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

A comparison of computation times with both 1 and 5 threads is given below.

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
#>              earth 536.0 540.0 552.5  546.0 566.5 577.8    10
#>  oumua (1 thread)  810.1 815.1 823.6  825.4 828.5 836.3    10
#>  oumua (5 threads) 200.9 206.2 213.2  209.2 211.2 241.6    10
```

Interaction Example with Factor
-------------------------------

We add a dummy variable to the model from before in the last example. The code is very similar.

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

The mean squared errors are given below.

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

A comparison of computation times with both 1 and 5 threads is given below.

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
#>              earth  910.6  915.0  950.4  947.2  978.4 1000.0    10
#>  oumua (1 thread)  1676.1 1687.4 1753.9 1715.0 1818.1 1897.2    10
#>  oumua (5 threads)  412.2  437.7  459.3  442.8  462.1  582.7    10
```

Comparison with the earth Package
---------------------------------

Some of the main differences between this package and the `earth` package is

-   the `earth` package has many more features!
-   the `earth` package is single threaded (as of this writing).
-   the `earth` includes the suggestions in Friedman (1993) to decrease the computation time. It would be nice to include these in this package. It only matters when there are many basis functions.
-   the `earth` package does not include an L2 penalty which simplifies the computation of the generalized cross validation criterion at each knot.
-   the `earth` package creates an orthogonal design matrix during the estimation (I think?) which allows one to skip some computations of the generalized cross validation criterion (a "full" forward and backward substitution).

Some computation can be skipped if one sets `lambda` to zero (i.e., no L2 penalty). The following code blocks shows the impact.

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
#>               expr    min     lq   mean median     uq  max neval
#>              earth  910.7  944.9  979.3  987.5 1015.3 1028    10
#>  oumua (1 thread)  1037.9 1058.1 1083.0 1091.1 1098.5 1139    10
#>  oumua (5 threads)  294.7  356.9  373.3  375.3  395.4  417    10
```

Settings `lambda = 0` yields one less back substitution for each knot position. Computing the orthogonal design matrix does come at a cost but it does seem worth it even with few knots. Though, I may though have misunderstood the C code in the `earth` package.

A final issue that still needs to be addressed is the L2 penalty in this package. When the knot position is found then the L2 penalty is applied to the coefficients in the model

![y = \\dots + \\beta\_1x\_i + \\beta\_2(x\_i - k)\_+](https://latex.codecogs.com/svg.latex?y%20%3D%20%5Cdots%20%2B%20%5Cbeta_1x_i%20%2B%20%5Cbeta_2%28x_i%20-%20k%29_%2B "y = \dots + \beta_1x_i + \beta_2(x_i - k)_+")

where ![k](https://latex.codecogs.com/svg.latex?k "k") is the knot and ![\\cdots](https://latex.codecogs.com/svg.latex?%5Ccdots "\cdots") are the other terms already included in the model. However, the final L2 penalty is applied to the coefficients in the model

![y = \\dots + \\beta\_1(k - x\_i)\_+ + \\beta\_2(x\_i - k)\_+](https://latex.codecogs.com/svg.latex?y%20%3D%20%5Cdots%20%2B%20%5Cbeta_1%28k%20-%20x_i%29_%2B%20%2B%20%5Cbeta_2%28x_i%20-%20k%29_%2B "y = \dots + \beta_1(k - x_i)_+ + \beta_2(x_i - k)_+")

which does not yield an equivalent model. The former is faster during the knot estimation while the latter is faster later as it yields a more sparse design matrix and thus faster computation times later.

References
----------

Friedman, Jerome H. 1991. “Multivariate Adaptive Regression Splines.” *Ann. Statist.* 19 (1). The Institute of Mathematical Statistics: 1–67. <https://doi.org/10.1214/aos/1176347963>.

———. 1993. “Fast Mars.” Technical Report 110. Stanford University Department of Statistics.
