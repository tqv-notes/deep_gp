# Deep Gaussian Process (MCMC) - C++ / Eigen port

A C++ translation of the Python prototype (`demo.py`, `dgp.py`, `mcmc.py`,
`tooling.py`) implementing a **two-layer Bayesian Deep Gaussian Process**
fitted by MCMC (Metropolis-within-Gibbs for the kernel hyper-parameters,
**elliptical slice sampling** for the latent layer).

Method follows Sauer, Gramacy & Higdon, *Active Learning for Deep Gaussian
Process Surrogates* (arXiv:2012.08015)

---

## Layout

```
dgp_cpp/
|-- include/dgp.hpp          # public API + structs
|-- src/
|   |-- tooling.cpp          # sq_dist, matern/exp2 kernels, Cholesky kriging
|   |-- mcmc.cpp             # logl, sample_g / sample_theta, ESS sample_w, gibbs
|   |-- dgp.cpp              # fit_dgp, predict_dgp (OpenMP-parallel)
|   |-- main.cpp             # demo driver (mirrors demo.py), writes CSVs
|-- scripts/plot_results.py  # reproduces the demo.py figures from the CSVs
|-- CMakeLists.txt
```

## Build

Requires a C++17 compiler and Eigen 3.3+ (OpenMP optional but recommended).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

and if you installed Eigen 3.3+ via vcpkg:

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/Users/quangvu197/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Without CMake you can compile directly:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 -Iinclude -I/path/to/eigen3 \
    src/tooling.cpp src/mcmc.cpp src/dgp.cpp src/main.cpp -o dgp_demo
```

## Run

```bash
build\Release\dgp_demo [nmcmc] [seed] [out_dir] [nchains]

build\Release\dgp_demo 1000 123 .\results 8 # multi-chain run

python scripts\plot_results.py .\results --save   # writes prediction.png, diagnostics.png
```

The demo writes `train.csv`, `truth.csv`, `pred.csv` (x, mean, variance),
`traces.csv` (MCMC traces), and `warp.csv` (latent warping samples).

---

## What changed vs. the Python prototype

### Performance (same algorithm, faster)

| stage   | Python (scipy) | C++ (1 chain) | speed-up |
|---------|----------------|---------------|----------|
| fit     | 18.6 s         | 0.17 s        | ~110×    |
| predict | 9.5 s          | 0.46 s        | ~20×     |

(`nmcmc = 1000`, `n = 50`, 200 test points, 2-core sandbox.)

The main drivers:

* **Cholesky instead of pseudo-inverse.** Every log-likelihood, sampler step
  and kriging call in the prototype built a full `scipy.linalg.pinv(K)` (an SVD)
  *and* a separate `det(K)`. For a symmetric positive-definite covariance the
  right tool is a single Cholesky factorisation: the quadratic form
  `xᵀK⁻¹x` becomes one triangular solve, and `log|K|` is `2·Σ log diag(L)` -
  free. This is the dominant cost and is both faster and more stable. The
  matrix inverse is never formed explicitly.
* **OpenMP** parallelism over MCMC draws in `predict_dgp`.
* Compiled, vectorised Eigen with `-O3 -march=native`.





