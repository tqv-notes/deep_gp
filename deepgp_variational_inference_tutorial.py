"""
doubly stochastic deep gaussian process (Salimbeni & Deisenroth, 2017)
from scratch in PyTorch. self-contained: no GPflow/GPyTorch.

model: y = f_L(...f_2(f_1(x))) + eps, each f_l a sparse GP.
inference: sample-through-the-layers variational inference.
ELBO = E_q[log p(y|f_L)] (Monte Carlo) - sum_l KL[q(u_l)||p(u_l)].
"""

import torch
import math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
JITTER = 1e-6

class RBFKernel(torch.nn.Module):
    """k(x,x') = s2 * exp(-|x-x'|^2 / (2 l^2)), ARD lengthscales."""

    def __init__(self, input_dim, lengthscale=1.0, variance=1.0):
        super().__init__()
        self.log_lengthscale = torch.nn.Parameter(
            math.log(lengthscale) * torch.ones(input_dim))
        self.log_variance = torch.nn.Parameter(torch.tensor(math.log(variance)))

    def K(self, X, X2=None):
        X2 = X if X2 is None else X2
        Xs = X / self.log_lengthscale.exp()
        X2s = X2 / self.log_lengthscale.exp()
        d2 = (Xs**2).sum(-1, keepdim=True) - 2 * Xs @ X2s.mT \
            + (X2s**2).sum(-1).unsqueeze(-2)
        return self.log_variance.exp() * torch.exp(-0.5 * d2.clamp_min(0.0))

    def K_diag(self, X):
        return self.log_variance.exp().expand(X.shape[:-1])


class SVGPLayer(torch.nn.Module):
    """sparse variational GP layer, whitened: q(v)=N(m, LL^T), u = chol(Kzz) v.

    inner layers use an identity/linear mean function (Salimbeni's trick) so
    that, at initialization, the layer behaves like the identity and gradients
    flow through a deep stack.
    """

    def __init__(self, kernel, Z, out_dim, mean_function=None):
        super().__init__()
        self.kernel = kernel
        M = Z.shape[0]
        self.Z = torch.nn.Parameter(Z.clone())
        self.q_mu = torch.nn.Parameter(torch.zeros(M, out_dim))
        self.q_sqrt = torch.nn.Parameter(
            1e-5 * torch.eye(M).expand(out_dim, M, M).clone())
        self.mean_function = mean_function  # None => zero mean

    def conditional(self, X):
        """q(f(X)) marginals. X: (..., N, D_in) -> mean, var: (..., N, D_out)."""
        Kzz = self.kernel.K(self.Z) + JITTER * torch.eye(len(self.Z))
        Lz = torch.linalg.cholesky(Kzz)
        Kzx = self.kernel.K(self.Z, X)                       # (..., M, N)
        A = torch.linalg.solve_triangular(Lz, Kzx, upper=False)  # whitened
        mean = A.mT @ self.q_mu                              # (..., N, D_out)
        SA = self.q_sqrt.mT @ A.unsqueeze(-3)                # (..., D_out, M, N)
        var = (self.kernel.K_diag(X).unsqueeze(-1)
               - (A**2).sum(-2, keepdim=True).mT
               + (SA**2).sum(-2).movedim(-2, -1))            # (..., N, D_out)
        if self.mean_function is not None:
            mean = mean + self.mean_function(X)
        return mean, var.clamp_min(1e-10)

    def sample(self, X):
        """reparameterized sample from q(f(X))."""
        mean, var = self.conditional(X)
        return mean + var.sqrt() * torch.randn_like(mean)

    def KL(self):
        """KL[q(v)||N(0,I)], summed over output dims."""
        L = torch.tril(self.q_sqrt)
        trace = (L**2).sum((-1, -2))
        logdet = torch.log(L.diagonal(dim1=-2, dim2=-1)**2).sum(-1)
        quad = (self.q_mu**2).sum(0)
        M = self.q_mu.shape[0]
        return 0.5 * (trace + quad - M - logdet).sum()


class DeepGP(torch.nn.Module):
    """stack of SVGP layers with a gaussian likelihood."""

    def __init__(self, layers, noise_var=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.log_noise = torch.nn.Parameter(torch.tensor(math.log(noise_var)))

    def propagate(self, X, num_samples=5):
        """draw S samples of the final-layer marginals q(f_L)."""
        F = X.expand(num_samples, *X.shape)
        for layer in self.layers[:-1]:
            F = layer.sample(F)
        return self.layers[-1].conditional(F)  # mean, var: (S, N, D_out)

    def elbo(self, X, y, num_samples=5, num_data=None):
        num_data = num_data if num_data is not None else len(X)
        mean, var = self.propagate(X, num_samples)
        s2 = self.log_noise.exp()
        # E_{q(f)}[log N(y|f, s2)], Gaussian case in closed form per sample:
        lik = (-0.5 * math.log(2 * math.pi) - 0.5 * s2.log()
               - 0.5 * ((y - mean)**2 + var) / s2)
        lik = lik.mean(0).sum()                       # avg samples, sum data
        kl = sum(layer.KL() for layer in self.layers)
        return (num_data / len(X)) * lik - kl

    @torch.no_grad()
    def predict(self, X, num_samples=100):
        """moments of the predictive mixture p(y*) ~ (1/S) sum_s N(m_s, v_s+s2)."""
        mean, var = self.propagate(X, num_samples)
        var = var + self.log_noise.exp()
        m = mean.mean(0)
        v = (var + mean**2).mean(0) - m**2
        return m, v


def make_dgp(dims, Z, lengthscale=1.0):
    """dims e.g. [1, 1, 1] => 2-layer DGP. inner layers get identity mean."""
    layers = []
    for l, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        last = l == len(dims) - 2
        mean_fn = None if last else (lambda x: x)  # identity (assumes d_in==d_out)
        kern = RBFKernel(d_in, lengthscale=lengthscale)
        layers.append(SVGPLayer(kern, Z.clone(), d_out, mean_function=mean_fn))
    return DeepGP(layers)


if __name__ == "__main__":
    # toy problem: a step function, hard for a (stationary) shallow GP
    N = 200
    X = torch.rand(N, 1) * 2 - 1
    y = (X > 0).double() + 0.05 * torch.randn(N, 1)

    Z = torch.linspace(-1, 1, 25).unsqueeze(-1)
    model = make_dgp([1, 1, 1], Z)  # 2-layer DGP

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(2000):
        opt.zero_grad()
        loss = -model.elbo(X, y, num_samples=5)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"step {step:4d}  ELBO {-loss.item():9.2f}")

    Xt = torch.linspace(-1.5, 1.5, 300).unsqueeze(-1)
    m, v = model.predict(Xt)
    print("noise std:", model.log_noise.exp().sqrt().item())
        
    flat = lambda t: t.squeeze(-1).tolist()
    plt.figure(figsize=(8, 4))
    plt.scatter(flat(X), flat(y), s=8, c="k", alpha=0.4, label="data")
    plt.plot(flat(Xt), flat(m), "C0", label="DGP mean")
    lo, hi = m - 2 * v.sqrt(), m + 2 * v.sqrt()
    plt.fill_between(flat(Xt), flat(lo), flat(hi), alpha=0.25, label="±2$\sigma$")
    plt.legend(); 
    plt.tight_layout()
    plt.show()