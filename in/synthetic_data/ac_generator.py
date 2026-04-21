import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


def generate_allen_cahn_data(visual_data=True):
    D = 1e-4
    L = 2.0
    Nx = 49
    Nt = 201
    noise_std = 0.001
    save_path = Path(".")
    save_path.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1, 1, Nx, endpoint=True)
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(Nx, d=dx)

    def reaction_term(u):
        return 5 * u - 5 * u**3

    def allen_cahn_ode(t, u_hat):
        u = np.real(ifft(u_hat))
        r_u_hat = fft(reaction_term(u))
        du_hat_dt = -D * (k**2) * u_hat + r_u_hat
        return du_hat_dt

    def add_noise(u):
        return u + np.random.normal(0, noise_std, u.shape)

    u0_train = (x**2) * np.cos(np.pi * x)
    t_span_indis = [0, 1.0]
    Nt_total = 2 * (Nt - 1) + 1
    t_eval_indis = np.linspace(0, 1.0, Nt_total)

    sol_indis = solve_ivp(allen_cahn_ode, t_span_indis, fft(u0_train), t_eval=t_eval_indis, method="RK45")

    # Training data
    t_train = t_eval_indis[:201]
    u_train_true = np.real(ifft(sol_indis.y[:, :201], axis=0)).T
    u_train_noisy = add_noise(u_train_true)

    # In-dis test data
    u_indis_true = np.real(ifft(sol_indis.y[:, -1]))
    u_indis_noisy = add_noise(u_indis_true)

    # Out-dis test data
    u0_ood = np.sin(np.pi * x / 2.0)
    sol_ood = solve_ivp(allen_cahn_ode, [0, 0.5], fft(u0_ood), t_eval=[0.5], method="RK45")
    u_ood_true = np.real(ifft(sol_ood.y[:, 0]))
    u_ood_noisy = add_noise(u_ood_true)

    np.save(save_path / "x_train.npy", x)
    np.save(save_path / "t_train.npy", t_train)
    np.save(save_path / "u_train.npy", u_train_noisy)

    np.save(save_path / "x_train-test.npy", x)
    np.save(save_path / "u_train-test.npy", u_train_noisy[-1])

    np.save(save_path / "x_in-test.npy", x)
    np.save(save_path / "u_in-test.npy", u_indis_noisy)

    np.save(save_path / "x_out-test.npy", x)
    np.save(save_path / "u_out-test.npy", u_ood_noisy)

    # Visualization
    if visual_data:

        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 2)

        # Unnoisy training data
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(u_train_true.T, extent=[0, 0.5, -1, 1], origin="lower", aspect="auto", cmap="RdBu_r")
        ax1.set_title("Uncorrupted training field $u(x, t)$")
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$x$")
        fig.colorbar(im1, ax=ax1)

        # True training field
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(u_train_noisy.T, extent=[0, 0.5, -1, 1], origin="lower", aspect="auto", cmap="RdBu_r", vmin=im1.get_array().min(), vmax=im1.get_array().max())
        ax2.set_title(f"Corrupted true training field $u(x,t)$ with noise ~ $N(0, {noise_std})$")
        ax2.set_xlabel("$t$")
        ax2.set_ylabel("$x$")
        fig.colorbar(im2, ax=ax2)

        # Test snapshots with noise
        ax3 = fig.add_subplot(gs[1, :])

        # Training test
        ax3.plot(x, u_train_true[-1], label="True training ($t=0.5, u(t=0) = x^2 \\cos(\\pi x)$)", lw=1.5)
        ax3.scatter(x, u_train_noisy[-1], color="C0", s=25, label="Noisy training samples", alpha=0.6)

        # In-Dis Test (extrapolation in time)
        ax3.plot(x, u_indis_true, label="True in-dis ($t=1.0, u(t=0) = x^2\\cos(\\pi x)$)", lw=1.5)
        ax3.scatter(x, u_indis_noisy, color="C1", s=25, label="Noisy in-dis samples", alpha=0.6)

        # Out-Dis Test (generalization to new IC)
        ax3.plot(x, u_ood_true, label="True out-dis ($t=0.5, u(t=0) = \\sin(\\pi x/2)$)", lw=1.5)
        ax3.scatter(x, u_ood_noisy, color="C2", s=25, label="Noisy out-dis samples", alpha=0.6)

        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel("x")
        ax3.set_ylabel("u")
        ax3.set_title("Test sets")
        ax3.legend()
        ax3.grid(True, alpha=0.2)

        # Reaction term plot
        u_react = np.linspace(-1.0, 1.0, 100)
        r_u = reaction_term(u_react)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(u_react, r_u, label="$R(u) = 5u - 5u^3$", color="C3")
        ax4.set_xlabel("$u$")
        ax4.set_ylabel("$R(u)$")
        ax4.set_title("Reaction term")
        ax4.axhline(0, color="gray", lw=0.5, ls="--")
        ax4.set_xlim(-1.0, 1.0)
        ax4.legend()
        ax4.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path / "synthetic_data.jpg", dpi=300)
        plt.show()


if __name__ == "__main__":
    generate_allen_cahn_data(visual_data=True)
