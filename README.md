
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
# functions to fit models
earth_call <- function(sims)
  earth(y ~ ., data = sims, endspan = 1, minspan = 1, degree = 1, penalty = 2)
oumua_call <- function(sims)
  oumua(y ~ ., data = sims, control = oumua.control(
    minspan = 1L, endspan = 1L, degree = 1L, penalty = 2, lambda = 1))

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
#>                 earth    oumua
#> mean           14.444 1.399002
#> standard error  5.279 0.004503
#> 
#> $`200`
#>                 earth    oumua
#> mean           1.9802 1.183308
#> standard error 0.4897 0.002059
#> 
#> $`500`
#>                 earth     oumua
#> mean           1.5320 1.0843347
#> standard error 0.2431 0.0007977
```

``` r
library(microbenchmark)
```

``` r
addi_runtimes <- local({
  run_dat <- additiv_sim(10000, 10)  
  microbenchmark(
    earth = earth_call(run_dat), oumua = oumua_call(run_dat), 
    times = 10)
})
```

``` r
summary(addi_runtimes)
#>    expr    min     lq  mean median    uq   max neval
#> 1 earth  96.14  97.28 105.9  100.9 115.1 129.4    10
#> 2 oumua 425.84 429.80 445.8  448.7 452.5 479.8    10
```
