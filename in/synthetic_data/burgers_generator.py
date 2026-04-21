import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


np.random.seed(42)


def generate_burgers_data(visual_data=True):
    D = 0.01 / np.pi
    L = 2.0
    Nx = 49
    Nt = 201
    noise_std = 0.03
    save_path = Path(".")
    save_path.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1, 1, Nx, endpoint=False)
    dx = x[1] - x[0]

    def advection(u):
        return u

    def burgers(t, u):
        u_pad = np.pad(u, 1, mode='wrap')
        u_left = u_pad[:-2]
        u_right = u_pad[2:]

        # Upwind advection
        ux_left = (u - u_left) / dx
        ux_right = (u_right - u) / dx
        u_x = np.where(u > 0, ux_left, ux_right)

        u_xx = (u_right - 2 * u + u_left) / (dx**2)

        dudt = -advection(u) * u_x + D * u_xx
        return dudt

    def add_noise(u):
        return u + np.random.normal(0, noise_std, u.shape)

    u0_train = - np.sin(np.pi * x)
    t_span_indis = [0, 2.0]
    Nt_total = 2 * (Nt - 1) + 1
    t_eval_indis = np.linspace(0, 2.0, Nt_total)

    sol_indis = solve_ivp(burgers, t_span_indis, u0_train, t_eval=t_eval_indis, method='RK45')

    # Training data
    t_train = t_eval_indis[:201]
    u_train_true = sol_indis.y[:, :201].T
    u_train_noisy = add_noise(u_train_true)

    # In-dis test data
    u_indis_true = sol_indis.y[:, -1]
    u_indis_noisy = add_noise(u_indis_true)

    # Out-dis test data
    u0_ood = np.sin(np.pi * x)
    sol_ood = solve_ivp(burgers, [0, 1], u0_ood, t_eval=[0.5])
    u_ood_true = sol_ood.y[:, 0]
    u_ood_noisy = add_noise(u_ood_true)

    # full data
    t_full = t_eval_indis
    u_full = sol_indis.y.T
    u_adv = np.linspace(-1.0, 1.0, 100)
    true_adv = advection(u_adv)

    np.save(save_path / "t_full.npy", t_full)
    np.save(save_path / "u_full.npy", u_full)
    np.save(save_path / "u_adv.npy", u_adv)
    np.save(save_path / "true_adv.npy", true_adv)

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
        im1 = ax1.imshow(u_train_true.T, extent=[0, 1, -1, 1], origin="lower", aspect="auto", cmap="RdBu_r")
        ax1.set_title("Uncorrupted training field $u(x, t)$")
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$x$")
        fig.colorbar(im1, ax=ax1)

        # True training field
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(u_train_noisy.T, extent=[0, 1, -1, 1], origin="lower", aspect="auto", cmap="RdBu_r", vmin=im1.get_array().min(), vmax=im1.get_array().max())
        ax2.set_title(f"Corrupted true training field $u(x,t)$ with noise ~ $N(0, {noise_std})$")
        ax2.set_xlabel("$t$")
        ax2.set_ylabel("$x$")
        fig.colorbar(im2, ax=ax2)

        # Test snapshots with noise
        ax3 = fig.add_subplot(gs[1, :])

        # Training test
        ax3.plot(x, u_train_true[-1], label="True training ($t=1.0, u(t=0) = -\\sin(\\pi x)$)", lw=1.5)
        ax3.scatter(x, u_train_noisy[-1], color="C0", s=25, label="Noisy training samples", alpha=0.6)

        # In-dis test (extrapolation in time)
        ax3.plot(x, u_indis_true, label="True in-dis ($t=2.0, u(t=0) = -\\sin(\\pi x)$)", lw=1.5)
        ax3.scatter(x, u_indis_noisy, color="C1", s=25, label="Noisy in-dis samples", alpha=0.6)

        # Out-dis test (generalization to new IC)
        ax3.plot(x, u_ood_true, label="True out-dis ($t=1.0, u(t=0) = \\sin(\\pi x)$)", lw=1.5)
        ax3.scatter(x, u_ood_noisy, color="C2", s=25, label="Noisy out-dis samples", alpha=0.6)

        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel("x")
        ax3.set_ylabel("u")
        ax3.set_title("Test sets")
        ax3.legend()
        ax3.grid(True, alpha=0.2)

        # Advection plot
        u_adv = np.linspace(-1.0, 1.0, 100)
        adv_u = advection(u_adv)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(u_adv, adv_u, label="$v(u) = u$", color="C3")
        ax4.set_xlabel("$u$")
        ax4.set_ylabel("$v(u)$")
        ax4.set_title("Advection velocity")
        ax4.axhline(0, color="gray", lw=0.5, ls="--")
        ax4.set_xlim(-1.0, 1.0)
        ax4.legend()
        ax4.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path / "synthetic_data.jpg", dpi=300)
        plt.show()


if __name__ == "__main__":
    generate_burgers_data(visual_data=True)
