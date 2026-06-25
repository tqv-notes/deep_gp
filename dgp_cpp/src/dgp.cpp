// top-level fit / predict for the two-layer Deep GP.
#include "dgp.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dgp {

DGPModel fit_dgp(const MatrixXd& x, const VectorXd& y, int nmcmc, int D,
                 bool verb, double g_0, double theta_y_0, double theta_w_0,
                 const double* true_g, double v, std::mt19937_64& rng) {
    DGPModel m;
    m.x = x;
    m.y = y;
    m.nmcmc = nmcmc;
    m.D = D;
    m.v = v;
    m.settings = Settings{};

    // w_0 = np.tile(x, [1, D]); for a single-column design this repeats x.
    const int n = static_cast<int>(x.rows());
    const int p = static_cast<int>(x.cols());
    Initial initial;
    initial.w = MatrixXd(n, D);
    for (int i = 0; i < D; ++i) initial.w.col(i) = x.col(i % p);
    initial.theta_y = theta_y_0;
    initial.theta_w = theta_w_0;
    initial.g = g_0;
    initial.tau2 = 1.0;

    m.samples = gibbs_sampling(x, y, nmcmc, D, verb, initial, true_g,
                               m.settings, v, rng);
    return m;
}

void predict_dgp(DGPModel& m, const MatrixXd& x_new) {
    m.x_new = x_new;
    const int n_new = static_cast<int>(x_new.rows());
    const int nmcmc = m.nmcmc;
    const int D = static_cast<int>(m.samples.w[0].cols());
    const double v = m.v;
    const double inner_tau2 = m.settings.inner_tau2;

    const MatrixXd dx = sq_dist(m.x);
    const MatrixXd d_cross = sq_dist(x_new, m.x);   // (n_new x n)

    MatrixXd mu_t(n_new, nmcmc);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<MatrixXd> sigma_local(nthreads,
                                      MatrixXd::Zero(n_new, n_new));

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < nmcmc; ++t) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const MatrixXd& w_t = m.samples.w[t];

        // layer 1: map the test inputs through each latent dimension.
        MatrixXd w_new(n_new, D);
        for (int i = 0; i < D; ++i) {
            KrigResult k = krig(w_t.col(i), dx, nullptr, &d_cross,
                                m.samples.theta_w(t, i), sqeps(), inner_tau2,
                                false, false, v);
            w_new.col(i) = k.mean;
        }

        // layer 2: GP on the latent layer -> predictive mean & covariance.
        MatrixXd dwt = sq_dist(w_t);
        MatrixXd dnew = sq_dist(w_new);
        MatrixXd dcr = sq_dist(w_new, w_t);
        KrigResult k = krig(m.y, dwt, &dnew, &dcr,
                            m.samples.theta_y(t), m.samples.g(t),
                            m.samples.tau2(t), false, true, v);
        mu_t.col(t) = k.mean;
        sigma_local[tid] += k.sigma;
    }

    MatrixXd sigma_sum = MatrixXd::Zero(n_new, n_new);
    for (int th = 0; th < nthreads; ++th) sigma_sum += sigma_local[th];

    // posterior predictive mean and (law-of-total-variance) covariance.
    m.mean = mu_t.rowwise().mean();
    MatrixXd centered = mu_t.colwise() - m.mean;            // (n_new x nmcmc)
    MatrixXd mu_cov = (centered * centered.transpose()) / (nmcmc - 1.0);
    m.sigma = sigma_sum / static_cast<double>(nmcmc) + mu_cov;
    m.has_prediction = true;
}

} // namespace dgp
