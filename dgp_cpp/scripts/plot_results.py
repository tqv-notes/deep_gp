"""
usage: python plot_results.py [results_dir] [--save]
  results_dir  directory containing train/truth/pred/traces/warp.csv (default .)
  --save       write PNGs instead of showing interactive windows
"""
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = "--save" in sys.argv
d = args[0] if args else "."
if save:
    matplotlib.use("Agg")

def load(name):
    return np.genfromtxt(os.path.join(d, name), delimiter=",", names=True)

train, truth, pred = load("train.csv"), load("truth.csv"), load("pred.csv")
traces = load("traces.csv")
warp = np.genfromtxt(os.path.join(d, "warp.csv"), delimiter=",", skip_header=1)

# MCMC diagnostics + warping samples
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].plot(traces["ll"]);      axs[0, 0].set_title("log-likelihood"); axs[0, 0].grid(True)
axs[0, 1].plot(traces["g"]);       axs[0, 1].set_title("g");              axs[0, 1].grid(True)
axs[1, 0].plot(traces["theta_y"]); axs[1, 0].set_title("theta_y");        axs[1, 0].grid(True)
axs[1, 1].plot(traces["theta_w"]); axs[1, 1].set_title("theta_w");        axs[1, 1].grid(True)
gs = axs[0, 2].get_gridspec()
for ax in axs[0:, -1]:
    ax.remove()
axbig = fig.add_subplot(gs[0:, -1])
xw = warp[:, 0]
for c in range(1, warp.shape[1]):
    axbig.plot(xw, warp[:, c], alpha=0.3)
axbig.set_xlabel("x"); axbig.set_ylabel("w"); axbig.set_title("warping samples"); axbig.grid(True)
plt.tight_layout()
if save:
    fig.savefig(os.path.join(d, "diagnostics.png"), dpi=110)

# predictions with 90% interval
fig2 = plt.figure(figsize=(10, 6))
plt.scatter(train["x"], train["y"], c="b", marker="o", label="training data")
plt.plot(truth["x"], truth["y"], "g.", label="true function")
plt.plot(pred["x"], pred["mean"], "r-", label="DGP mean")
sd = np.sqrt(np.maximum(pred["var"], 0.0))
lb = pred["mean"] + norm.ppf(0.05, 0, sd)
ub = pred["mean"] + norm.ppf(0.95, 0, sd)
plt.plot(pred["x"], lb, "m--", label="90% confidence interval")
plt.plot(pred["x"], ub, "m--")
plt.xlabel("x"); plt.ylabel("y"); plt.title("DGP with two layers")
plt.legend(); plt.grid(True)
plt.tight_layout()
if save:
    fig2.savefig(os.path.join(d, "prediction.png"), dpi=110)
    print("saved diagnostics.png and prediction.png to", d)
else:
    plt.show()
