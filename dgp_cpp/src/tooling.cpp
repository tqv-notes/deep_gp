// covariance kernels, squared distances, and Cholesky-based kriging.
#include "dgp.hpp"
#include <stdexcept>

namespace dgp {

MatrixXd sq_dist(const MatrixXd& X1, const MatrixXd& X2) {
    // D(i,j) = ||X1_i - X2_j||^2 = |X1_i|^2 + |X2_j|^2 - 2 X1_i . X2_j
    const VectorXd s1 = X1.rowwise().squaredNorm();   // (n1)
    const VectorXd s2 = X2.rowwise().squaredNorm();   // (n2)
    MatrixXd D = (-2.0) * (X1 * X2.transpose());      // (n1 x n2)
    D.colwise() += s1;             // add s1(i) to row i
    D.rowwise() += s2.transpose(); // add s2(j) to column j
    // clamp tiny negative round-off so sqrt() in the kernels stays real.
    return D.cwiseMax(0.0);
}

MatrixXd sq_dist(const MatrixXd& X1) { return sq_dist(X1, X1); }

MatrixXd matern(const MatrixXd& distmat, double tau2, double theta, double g, double v) {
    const Eigen::Index n1 = distmat.rows(), n2 = distmat.cols();
    MatrixXd cov(n1, n2);
    if (v == 0.5) {
        Eigen::ArrayXXd r = (distmat.array() / theta).sqrt();
        cov = (tau2 * (-r).exp()).matrix();
    } else if (v == 1.5) {
        Eigen::ArrayXXd r = (3.0 * distmat.array() / theta).sqrt();
        cov = (tau2 * (1.0 + r) * (-r).exp()).matrix();
    } else if (v == 2.5) {
        Eigen::ArrayXXd r = (5.0 * distmat.array() / theta).sqrt();
        cov = (tau2 * (1.0 + r + r * r / 3.0) * (-r).exp()).matrix();
    } else {
        throw std::invalid_argument("matern: v must be 0.5, 1.5 or 2.5");
    }
    if (n1 == n2)
        cov.diagonal().array() += tau2 * g;
    return cov;
}

MatrixXd exp2(const MatrixXd& distmat, double tau2, double theta, double g) {
    const Eigen::Index n1 = distmat.rows(), n2 = distmat.cols();
    MatrixXd cov = (tau2 * (-(distmat.array() / theta)).exp()).matrix();
    if (n1 == n2)
        cov.diagonal().array() += tau2 * g;
    return cov;
}

MatrixXd cov_fn(const MatrixXd& distmat, double tau2, double theta, double g, double v) {
    return (v == EXP2) ? exp2(distmat, tau2, theta, g)
                       : matern(distmat, tau2, theta, g, v);
}

// robust Cholesky: factorise K, adding escalating jitter if not positive definite. 
// returns the LLT factorisation.
Eigen::LLT<MatrixXd> robust_llt(MatrixXd K) {
    Eigen::LLT<MatrixXd> llt(K);
    if (llt.info() == Eigen::Success) return llt;
    double jitter = 1e-12 * K.diagonal().mean();
    for (int it = 0; it < 10; ++it) {
        MatrixXd Kj = K;
        Kj.diagonal().array() += jitter;
        llt.compute(Kj);
        if (llt.info() == Eigen::Success) return llt;
        jitter *= 10.0;
    }
    return llt; // caller proceeds; extremely unlikely to reach here
}

KrigResult krig(const VectorXd& y, const MatrixXd& dx,
                const MatrixXd* d_new, const MatrixXd* d_cross,
                double theta, double g, double tau2,
                bool want_s2, bool want_sigma, double v,
                double prior_mean, double prior_mean_new) {
    KrigResult out;

    MatrixXd C = cov_fn(dx, 1.0, theta, g, v);          // (n x n)
    Eigen::LLT<MatrixXd> llt = robust_llt(C);

    // C_cross = cov(new, design), nugget 0  -> (n_new x n)
    MatrixXd C_cross;
    if (d_cross)
        C_cross = cov_fn(*d_cross, 1.0, theta, 0.0, v);

    // alpha = C^{-1} (y - prior_mean)
    VectorXd ymm = y.array() - prior_mean;
    VectorXd alpha = llt.solve(ymm);
    out.mean = (C_cross * alpha).array() + prior_mean_new;

    if (want_s2 && d_new) {
        // V = L^{-1} C_cross^T  ->  diag(C_cross C^{-1} C_cross^T) = colwise V.^2
        MatrixXd V = llt.matrixL().solve(C_cross.transpose());   // (n x n_new)
        VectorXd qdiag = V.colwise().squaredNorm();              // (n_new)
        VectorXd Cnew = VectorXd::Constant(d_new->rows(), 1.0 + g);
        out.s2 = tau2 * (Cnew - qdiag);
        out.has_s2 = true;
    }
    if (want_sigma && d_new) {
        MatrixXd V = llt.matrixL().solve(C_cross.transpose());   // (n x n_new)
        MatrixXd quad = V.transpose() * V;                       // (n_new x n_new)
        MatrixXd Cnew = cov_fn(*d_new, 1.0, theta, g, v);
        out.sigma = tau2 * (Cnew - quad);
        out.has_sigma = true;
    }
    return out;
}

} // namespace dgp
