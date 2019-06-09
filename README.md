
Oumuamua
========

``` r
additiv_sim <- function(N, p){
  p <- max(5L, p)
  x <- matrix(runif(N * p), nc = p)
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
# returns minspan and endspan arguments. Similar to Friedman (1991) 
# though without ajdusting N for number of non-negative elements in basis 
# function.
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
oumua_call <- function(sims){
  spans <- get_spans(N = nrow(sims), p = p)
  oumua(y ~ ., data = sims, control = oumua.control(
    minspan = spans["minspan"], endspan = spans["endspan"], degree = 1L, 
    penalty = 2, lambda = 1))
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
#> mean           1.424117 1.404078
#> standard error 0.005947 0.004507
#> 
#> $`200`
#>                   earth    oumua
#> mean           1.205834 1.185529
#> standard error 0.002613 0.002049
#> 
#> $`500`
#>                    earth     oumua
#> mean           1.0930594 1.0845603
#> standard error 0.0008732 0.0007714
```

``` r
library(microbenchmark)
```

``` r
addi_runtimes <- local({
  run_dat <- additiv_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), oumua = oumua_call(run_dat), 
    times = 100)
})
```

``` r
addi_runtimes
#> Unit: milliseconds
#>   expr    min     lq   mean median     uq   max neval
#>  earth  77.33  79.99  86.16  82.08  86.15 141.5   100
#>  oumua 116.63 119.83 125.61 122.37 126.61 209.7   100
```
