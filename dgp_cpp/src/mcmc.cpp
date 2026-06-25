// log-likelihood, Metropolis-within-Gibbs samplers and the Gibbs/ESS loop.
#include "dgp.hpp"
#include <stdexcept>
#include <cstdio>

namespace dgp {

static const double TWO_PI = 6.283185307179586476925286766559;

// log of scipy.stats.gamma.pdf(x, alpha, scale=1/beta).
static double log_gamma_pdf(double x, double alpha, double beta) {
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    return alpha * std::log(beta) + (alpha - 1.0) * std::log(x)
           - beta * x - std::lgamma(alpha);
}

LoglResult logl(const VectorXd& out_vec, const MatrixXd& in_dmat,
                double g, double theta, bool outer, double v,
                bool want_tau2, double mu, double scale) {
    const double n = static_cast<double>(out_vec.size());
    MatrixXd K = scale * cov_fn(in_dmat, 1.0, theta, g, v);
    Eigen::LLT<MatrixXd> llt = robust_llt(K);

    VectorXd diff = out_vec.array() - mu;
    // quadterm = diff' K^{-1} diff = || L^{-1} diff ||^2
    VectorXd Linv_diff = llt.matrixL().solve(diff);
    double quadterm = Linv_diff.squaredNorm();
    // log|K| = 2 * sum(log(diag(L)))
    double logdet = 2.0 * llt.matrixLLT().diagonal().array().log().sum();

    LoglResult r;
    if (outer)
        r.logl = (-n * 0.5) * std::log(quadterm) - 0.5 * logdet;
    else
        r.logl = -0.5 * logdet - 0.5 * quadterm;

    if (want_tau2) { r.tau2 = quadterm / n; r.has_tau2 = true; }
    return r;
}

ScalarSample sample_g(const VectorXd& out_vec, const MatrixXd& in_dmat,
                      double g_t, double theta, double alpha, double beta,
                      double l, double u, const double* ll_prev, double v,
                      std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double lo = l * g_t / u, hi = u * g_t / l;
    double g_star = lo + (hi - lo) * U(rng);
    double ru = U(rng);

    double llp = ll_prev ? *ll_prev
                         : logl(out_vec, in_dmat, g_t, theta, true, v).logl;

    double thr = llp + log_gamma_pdf(g_t - sqeps(), alpha, beta)
                 + std::log(ru) - std::log(g_t) + std::log(g_star);

    double ll_new = logl(out_vec, in_dmat, g_star, theta, true, v).logl;
    double cand = ll_new + log_gamma_pdf(g_star - sqeps(), alpha, beta);

    if (cand > thr) return {g_star, ll_new};
    return {g_t, llp};
}

ThetaSample sample_theta(const VectorXd& out_vec, const MatrixXd& in_dmat,
                         double g, double theta_t, double alpha, double beta,
                         double l, double u, bool outer, const double* ll_prev,
                         double v, bool want_tau2, double prior_mean, double scale,
                         std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double lo = l * theta_t / u, hi = u * theta_t / l;
    double theta_star = lo + (hi - lo) * U(rng);
    double ru = U(rng);

    double llp = ll_prev ? *ll_prev
                         : logl(out_vec, in_dmat, g, theta_t, outer, v, false,
                                prior_mean, scale).logl;

    double thr = llp + log_gamma_pdf(theta_t - sqeps(), alpha, beta)
                 + std::log(ru) - std::log(theta_t) + std::log(theta_star);

    LoglResult ln = logl(out_vec, in_dmat, g, theta_star, outer, v, want_tau2,
                         prior_mean, scale);
    double cand = ln.logl + log_gamma_pdf(theta_star - sqeps(), alpha, beta);

    if (cand > thr)
        return {theta_star, ln.logl, ln.has_tau2 ? ln.tau2 : 0.0, ln.has_tau2};
    return {theta_t, llp, 0.0, false};
}

// draw a zero-mean multivariate normal with covariance `cov` (Cholesky).
static VectorXd mvn_zero(const MatrixXd& cov, std::mt19937_64& rng) {
    Eigen::LLT<MatrixXd> llt = robust_llt(cov);
    VectorXd z(cov.rows());
    std::normal_distribution<double> N(0.0, 1.0);
    for (Eigen::Index i = 0; i < z.size(); ++i) z(i) = N(rng);
    return llt.matrixL() * z;
}

WSample sample_w(const VectorXd& out_vec, MatrixXd& wt, const MatrixXd& wt_dmat,
                 const MatrixXd& in_dmat, double g, double theta_y,
                 const VectorXd& theta_w, const double* ll_prev, double v,
                 double scale, std::mt19937_64& rng) {
    const int D = static_cast<int>(wt.cols());
    std::uniform_real_distribution<double> U(0.0, 1.0);

    double llp = ll_prev ? *ll_prev
                         : logl(out_vec, wt_dmat, g, theta_y, true, v).logl;

    MatrixXd dw = wt_dmat;
    for (int i = 0; i < D; ++i) {
        MatrixXd cov = scale * cov_fn(in_dmat, 1.0, theta_w(i), 0.0, v);
        VectorXd w_prior = mvn_zero(cov, rng);

        double a = TWO_PI * U(rng);
        double amin = a - TWO_PI, amax = a;
        double ru = U(rng);
        double thr = llp + std::log(ru);

        bool accept = false;
        int count = 0;
        VectorXd w_prev = wt.col(i);  // copy

        while (!accept) {
            ++count;
            wt.col(i) = w_prev * std::cos(a) + w_prior * std::sin(a);
            dw = sq_dist(wt);
            double nl = logl(out_vec, dw, g, theta_y, true, v).logl;
            if (nl > thr) {
                llp = nl;
                accept = true;
            } else {
                if (a < 0.0) amin = a; else amax = a;
                a = amin + (amax - amin) * U(rng);
                if (count > 100)
                    throw std::runtime_error("reached maximum iteration of ESS");
            }
        }
    }
    return {llp, dw};
}

GibbsResult gibbs_sampling(const MatrixXd& x, const VectorXd& y, int nmcmc, int D,
                           bool verb, const Initial& initial, const double* true_g,
                           const Settings& s, double v, std::mt19937_64& rng) {
    const int n = static_cast<int>(x.rows());
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    MatrixXd dx = sq_dist(x);
    MatrixXd dw = sq_dist(initial.w);

    GibbsResult R;
    R.g = VectorXd::Constant(nmcmc, NaN);
    R.theta_y = VectorXd::Constant(nmcmc, NaN);
    R.theta_w = MatrixXd::Constant(nmcmc, D, NaN);
    R.tau2 = VectorXd::Constant(nmcmc, NaN);
    R.ll = VectorXd::Constant(nmcmc, NaN);
    R.w.assign(nmcmc, MatrixXd());

    R.g(0) = true_g ? *true_g : initial.g;
    R.theta_y(0) = initial.theta_y;
    R.theta_w.row(0).setConstant(initial.theta_w);
    R.w[0] = initial.w;
    R.tau2(0) = initial.tau2;

    bool have_ll = false;
    double ll_outer = 0.0;

    for (int j = 1; j < nmcmc; ++j) {
        if (verb && (j + 1) % 500 == 0) { std::printf("%d\n", j + 1); std::fflush(stdout); }

        // nugget g (outer layer)
        if (!true_g) {
            ScalarSample sg = sample_g(y, dw, R.g(j - 1), R.theta_y(j - 1),
                                       s.alpha_g, s.beta_g, s.l, s.u,
                                       have_ll ? &ll_outer : nullptr, v, rng);
            R.g(j) = sg.value;
            ll_outer = sg.ll; have_ll = true;
        } else {
            R.g(j) = *true_g;
        }

        // lengthscale theta_y (also yields tau2)
        ThetaSample sty = sample_theta(y, dw, R.g(j), R.theta_y(j - 1),
                                       s.alpha_theta_y, s.beta_theta_y, s.l, s.u,
                                       true, have_ll ? &ll_outer : nullptr, v,
                                       true, 0.0, 1.0, rng);
        R.theta_y(j) = sty.theta;
        ll_outer = sty.ll; have_ll = true;
        R.tau2(j) = sty.has_tau2 ? sty.tau2 : R.tau2(j - 1);

        // inner lengthscales theta_w (per latent dimension)
        const MatrixXd& w_prev_mat = R.w[j - 1];
        for (int i = 0; i < D; ++i) {
            ThetaSample stw = sample_theta(w_prev_mat.col(i), dx, sqeps(),
                                           R.theta_w(j - 1, i), s.alpha_theta_w,
                                           s.beta_theta_w, s.l, s.u, false,
                                           nullptr, v, false, 0.0, s.inner_tau2, rng);
            R.theta_w(j, i) = stw.theta;
        }

        // latent layer W via elliptical slice sampling
        MatrixXd w_cur = R.w[j - 1];  // modified in place
        VectorXd theta_w_row = R.theta_w.row(j).transpose();
        WSample sw = sample_w(y, w_cur, dw, dx, R.g(j), R.theta_y(j),
                              theta_w_row, &ll_outer, v, s.inner_tau2, rng);
        R.w[j] = w_cur;
        ll_outer = sw.ll; have_ll = true;
        R.ll(j) = ll_outer;
        dw = sw.dw;
    }
    (void)n;
    return R;
}

} // namespace dgp
