// Demo driver: mirrors demo.py. Fits a 2-layer Deep GP to a synthetic
// function, predicts on a dense grid, and writes CSVs for plotting.
//
// Usage: dgp_demo [nmcmc] [seed] [out_dir] [nchains]
//   nmcmc    number of MCMC iterations           (default 1000)
//   seed     base RNG seed                       (default 123)
//   out_dir  directory for CSV output            (default ".")
//   nchains  independent restarts; the chain with the highest mean
//            second-half log-likelihood is kept  (default 1)

#include "dgp.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>

using namespace dgp;

static double f(double x) {
    const double pi = 3.14159265358979323846;
    return std::sin(2.0 * pi * 2.0 * (1.0 + 2.0 * x * x) * x);
}

static VectorXd linspace(double a, double b, int n) {
    VectorXd v(n);
    if (n == 1) { v(0) = a; return v; }
    for (int i = 0; i < n; ++i) v(i) = a + (b - a) * i / (n - 1);
    return v;
}

static void write_csv(const std::string& path, const std::string& header,
                      const MatrixXd& data) {
    std::ofstream o(path);
    o << header << "\n";
    o.precision(12);
    for (Eigen::Index i = 0; i < data.rows(); ++i) {
        for (Eigen::Index j = 0; j < data.cols(); ++j) {
            o << data(i, j);
            if (j + 1 < data.cols()) o << ",";
        }
        o << "\n";
    }
}

// mean log-likelihood over the second half of the chain (mixing score).
static double chain_score(const DGPModel& m) {
    int lo = m.nmcmc / 2;
    double s = 0.0; int c = 0;
    for (int j = lo; j < m.nmcmc; ++j) {
        double v = m.samples.ll(j);
        if (std::isfinite(v)) { s += v; ++c; }
    }
    return c ? s / c : -std::numeric_limits<double>::infinity();
}

int main(int argc, char** argv) {
    int nmcmc = (argc > 1) ? std::stoi(argv[1]) : 1000;
    unsigned long seed = (argc > 2) ? std::stoul(argv[2]) : 123UL;
    std::string out = (argc > 3) ? argv[3] : ".";
    int nchains = (argc > 4) ? std::stoi(argv[4]) : 1;
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    if (!out.empty() && out.back() != '/' && out.back() != '\\') out += "/";

    const int n = 50, n_test = 200;
    VectorXd xv = linspace(0.0, 1.0, n);
    MatrixXd x = xv;
    VectorXd y(n);
    for (int i = 0; i < n; ++i) y(i) = f(xv(i));

    VectorXd xpv = linspace(0.0, 1.0, n_test);
    MatrixXd xp = xpv;
    VectorXd yp(n_test);
    for (int i = 0; i < n_test; ++i) yp(i) = f(xpv(i));

    std::cout << "fitting Deep GP: nmcmc=" << nmcmc << " seed=" << seed
              << " n=" << n << " chains=" << nchains << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    DGPModel best;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < nchains; ++c) {
        std::mt19937_64 rng(seed + static_cast<unsigned long>(c));
        DGPModel m = fit_dgp(x, y, nmcmc, /*D=*/1, /*verb=*/(nchains == 1),
                             /*g_0=*/0.01, /*theta_y_0=*/0.1, /*theta_w_0=*/0.1,
                             /*true_g=*/nullptr, /*v=*/2.5, rng);
        double sc = chain_score(m);
        if (nchains > 1)
            std::cout << "  chain " << c << " (seed " << (seed + c)
                      << ") mean-ll=" << sc << "\n";
        if (sc > best_score) { best_score = sc; best = std::move(m); }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    predict_dgp(best, xp);
    auto t2 = std::chrono::high_resolution_clock::now();

    double fit_s = std::chrono::duration<double>(t1 - t0).count();
    double pred_s = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "fit:     " << fit_s << " s (" << nchains << " chain(s))\n";
    std::cout << "predict: " << pred_s << " s\n";
    std::cout << "kept chain mean-ll: " << best_score << "\n";

    double rmse = std::sqrt((best.mean - yp).squaredNorm() / n_test);
    std::cout << "test RMSE (mean vs truth): " << rmse << "\n";

    MatrixXd train(n, 2); train.col(0) = xv; train.col(1) = y;
    write_csv(out + "train.csv", "x,y", train);

    MatrixXd truth(n_test, 2); truth.col(0) = xpv; truth.col(1) = yp;
    write_csv(out + "truth.csv", "x,y", truth);

    VectorXd var = best.sigma.diagonal();
    MatrixXd pred(n_test, 3);
    pred.col(0) = xpv; pred.col(1) = best.mean; pred.col(2) = var;
    write_csv(out + "pred.csv", "x,mean,var", pred);

    MatrixXd tr(nmcmc, 5);
    for (int j = 0; j < nmcmc; ++j) {
        tr(j, 0) = j;
        tr(j, 1) = best.samples.ll(j);
        tr(j, 2) = best.samples.g(j);
        tr(j, 3) = best.samples.theta_y(j);
        tr(j, 4) = best.samples.theta_w(j, 0);
    }
    write_csv(out + "traces.csv", "iter,ll,g,theta_y,theta_w", tr);

    int step = std::max(1, (int)std::lround((double)nmcmc / 1000.0));
    std::vector<int> idxs;
    for (int j = 0; j < nmcmc; j += step) idxs.push_back(j);
    MatrixXd warp(n, 1 + (int)idxs.size());
    warp.col(0) = xv;
    for (size_t c = 0; c < idxs.size(); ++c)
        warp.col(1 + (int)c) = best.samples.w[idxs[c]].col(0);
    std::string wh = "x";
    for (int it : idxs) wh += ",w" + std::to_string(it);
    write_csv(out + "warp.csv", wh, warp);

    std::cout << "wrote train.csv truth.csv pred.csv traces.csv warp.csv to\n  "
              << std::filesystem::absolute(out).string() << "\n";
    return 0;
}