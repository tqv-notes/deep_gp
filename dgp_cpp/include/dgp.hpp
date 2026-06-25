#pragma once
//
// bayesian two-layer Deep Gaussian Process via MCMC (C++/Eigen port).
//

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

namespace dgp {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// covariance selector: Matern smoothness must be one of {0.5, 1.5, 2.5};
// the sentinel EXP2 selects the squared-exponential ("exp2") kernel.
constexpr double EXP2 = -1.0;

inline double sqeps() {
    // sqrt(machine epsilon): the small nugget used for the inner layer
    static const double v = std::sqrt(std::numeric_limits<double>::epsilon());
    return v;
}

// ========================== tooling =======================================

// pairwise squared Euclidean distance matrix. sq_dist(X) is the symmetric
// self-distance; sq_dist(X1, X2) is the (n1 x n2) cross-distance.
MatrixXd sq_dist(const MatrixXd& X1);
MatrixXd sq_dist(const MatrixXd& X1, const MatrixXd& X2);

// covariance matrices built from a squared-distance matrix `distmat`.
// `g` is the nugget added to the diagonal (only when the matrix is square).
MatrixXd matern(const MatrixXd& distmat, double tau2, double theta, double g, double v);
MatrixXd exp2(const MatrixXd& distmat, double tau2, double theta, double g);
// dispatch on the covariance selector v (EXP2 -> exp2, else matern).
MatrixXd cov_fn(const MatrixXd& distmat, double tau2, double theta, double g, double v);

// cholesky factorisation with escalating jitter fallback for (near-)singular SPD matrices.
Eigen::LLT<MatrixXd> robust_llt(MatrixXd K);

struct KrigResult {
    VectorXd mean;
    VectorXd s2;        // populated when want_s2
    MatrixXd sigma;     // populated when want_sigma
    bool has_s2 = false;
    bool has_sigma = false;
};

// kriging (GP conditional mean / covariance). `dx` is sq_dist of the design,
// `d_cross` = sq_dist(new, design), `d_new` = sq_dist(new). Cholesky-based.
KrigResult krig(const VectorXd& y, const MatrixXd& dx,
                const MatrixXd* d_new, const MatrixXd* d_cross,
                double theta, double g, double tau2,
                bool want_s2, bool want_sigma, double v,
                double prior_mean = 0.0, double prior_mean_new = 0.0);

// ========================== mcmc =========================================

struct LoglResult {
    double logl;
    double tau2 = 0.0;
    bool has_tau2 = false;
};

// profile / concentrated log-likelihood of a GP layer (Cholesky based).
LoglResult logl(const VectorXd& out_vec, const MatrixXd& in_dmat,
                double g, double theta, bool outer, double v,
                bool want_tau2 = false, double mu = 0.0, double scale = 1.0);

struct ScalarSample { double value; double ll; };
struct ThetaSample { double theta; double ll; double tau2; bool has_tau2; };

// metropolis-within-Gibbs updates for the nugget g and lengthscale theta.
ScalarSample sample_g(const VectorXd& out_vec, const MatrixXd& in_dmat,
                      double g_t, double theta, double alpha, double beta,
                      double l, double u, const double* ll_prev, double v,
                      std::mt19937_64& rng);

ThetaSample sample_theta(const VectorXd& out_vec, const MatrixXd& in_dmat,
                         double g, double theta_t, double alpha, double beta,
                         double l, double u, bool outer, const double* ll_prev,
                         double v, bool want_tau2, double prior_mean, double scale,
                         std::mt19937_64& rng);

struct WSample { double ll; MatrixXd dw; };

// elliptical slice sampling of the latent layer W (updated in place).
WSample sample_w(const VectorXd& out_vec, MatrixXd& wt, const MatrixXd& wt_dmat,
                 const MatrixXd& in_dmat, double g, double theta_y,
                 const VectorXd& theta_w, const double* ll_prev, double v,
                 double scale, std::mt19937_64& rng);

struct Settings {
    double l = 1.0, u = 2.0;
    double alpha_g = 1.5, alpha_theta_w = 1.5, alpha_theta_y = 1.5;
    double beta_g = 3.9, beta_theta_w = 3.9 / 4.0, beta_theta_y = 3.9 / 6.0;
    double inner_tau2 = 1.0;
};

struct Initial {
    MatrixXd w;
    double theta_y = 0.1;
    double theta_w = 0.1;
    double g = 0.01;
    double tau2 = 1.0;
};

struct GibbsResult {
    VectorXd g;               // (nmcmc)
    VectorXd theta_y;         // (nmcmc)
    MatrixXd theta_w;         // (nmcmc x D)
    std::vector<MatrixXd> w;  // nmcmc entries, each (n x D)
    VectorXd tau2;            // (nmcmc)
    VectorXd ll;              // (nmcmc)
};

GibbsResult gibbs_sampling(const MatrixXd& x, const VectorXd& y, int nmcmc, int D,
                           bool verb, const Initial& initial, const double* true_g,
                           const Settings& settings, double v,
                           std::mt19937_64& rng);

// ========================== dgp ================================================

struct DGPModel {
    MatrixXd x;
    VectorXd y;
    int nmcmc = 0;
    int D = 0;
    double v = 2.5;
    Settings settings;
    GibbsResult samples;

    // prediction outputs (filled by predict_dgp).
    MatrixXd x_new;
    VectorXd mean;
    MatrixXd sigma;
    bool has_prediction = false;
};

DGPModel fit_dgp(const MatrixXd& x, const VectorXd& y, int nmcmc, int D,
                 bool verb, double g_0, double theta_y_0, double theta_w_0,
                 const double* true_g, double v, std::mt19937_64& rng);

void predict_dgp(DGPModel& model, const MatrixXd& x_new);

} // namespace dgp
