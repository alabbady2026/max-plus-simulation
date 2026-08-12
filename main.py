import numpy as np
import matplotlib.pyplot as plt

# Max-Plus Negative Infinity (\epsilon)
EPS = -np.inf


def maxplus_matmul(A, B):
    """Computes C = A \otimes B in Max-Plus Algebra."""
    n, p1 = A.shape
    p2, m = B.shape
    assert p1 == p2, f"Dimension mismatch: {A.shape} vs {B.shape}"

    C = np.full((n, m), EPS)
    for i in range(n):
        for j in range(m):
            # C_ij = \bigoplus_k (A_ik \otimes B_kj) = \max_k (A_ik + B_kj)
            C[i, j] = np.max(A[i, :] + B[:, j])
    return C


class TEGSimulator:
    """Timed Event Graph (TEG) State-Space Simulator using Max-Plus Algebra."""

    def __init__(self, A, B, C):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.n = self.A.shape[0]

    def simulate(self, u_trajectory, x0=None):
        """Simulates state and output trajectories given input schedule u_trajectory."""
        K = u_trajectory.shape[1]
        x_history = np.full((self.n, K), EPS)
        y_history = np.full((self.C.shape[0], K), EPS)

        x_prev = (
            np.zeros((self.n, 1))
            if x0 is None
            else np.array(x0, dtype=float).reshape((self.n, 1))
        )

        for k in range(K):
            u_k = u_trajectory[:, k].reshape((-1, 1))

            # State equation: x(k) = A \otimes x(k-1) \oplus B \otimes u(k)
            Ax = maxplus_matmul(self.A, x_prev)
            Bu = maxplus_matmul(self.B, u_k)
            x_k = np.maximum(Ax, Bu)

            # Output equation: y(k) = C \otimes x(k)
            y_k = maxplus_matmul(self.C, x_k)

            x_history[:, k] = x_k.flatten()
            y_history[:, k] = y_k.flatten()
            x_prev = x_k

        return x_history, y_history


def main():
    # 1. Define System Matrices from Section 5 Case Study
    A = [
        [3, EPS, EPS, EPS],
        [EPS, 4, EPS, EPS],
        [5, 5, EPS, 3],
        [10, 10, EPS, 8],
    ]

    # B = [[3, EPS], [EPS, 4], [EPS, EPS], [EPS, EPS]]
    B = [[0, EPS], [EPS, 0], [2, 1], [7, 6]]

    C = [[9, 8, 7, 2]]

    # 2. Input trajectory: u_1(k) = 5k, u_2(k) = 5k for k = 1..6
    K = 6
    k_steps = np.arange(1, K + 1)
    u_trajectory = np.array([5 * (k_steps-1), 5 * (k_steps-1)], dtype=float)

    # 3. Instantiate and run simulation
    sim = TEGSimulator(A, B, C)
    x_hist, y_hist = sim.simulate(u_trajectory)

    # 4. Print results table
    print("=" * 65)
    print("SIMULATION RESULTS TABLE")
    print("=" * 65)
    print(f"{'Event Step (k)':<20}" + "".join([f"k={k:<6}" for k in k_steps]))
    print("-" * 65)
    print(
        f"{'u1(k)':<20}"
        + "".join([f"{val:<8.0f}" for val in u_trajectory[0, :]])
    )
    print(
        f"{'u2(k)':<20}"
        + "".join([f"{val:<8.0f}" for val in u_trajectory[1, :]])
    )
    print("-" * 65)
    for i in range(4):
        print(
            f"{f'x{i+1}(k)':<20}"
            + "".join([f"{val:<8.0f}" for val in x_hist[i, :]])
        )
    print("-" * 65)
    print(
        f"{'y(k) Output':<20}"
        + "".join([f"{val:<8.0f}" for val in y_hist[0, :]])
    )
    print("=" * 65)

    # 5. Plotting results
    plt.figure(figsize=(8, 5))
    plt.plot(k_steps, u_trajectory[0, :], "g--s", label="Inputs u(k)")
    plt.plot(k_steps, x_hist[0, :], "b-^", label="x1(k) (Line A)")
    plt.plot(k_steps, x_hist[1, :], "c-d", label="x2(k) (Line B)")
    plt.plot(k_steps, x_hist[2, :], "m-p", label="x3(k) (Assembly)")
    plt.plot(k_steps, x_hist[3, :], "y-h", label="x4(k) (AGV)")
    plt.plot(k_steps, y_hist[0, :], "r-o", linewidth=2, label="y(k) (Output)")

    plt.title("Event Firing Trajectories and Bottleneck Dynamics")
    plt.xlabel("Event Step (k)")
    plt.ylabel("Time Date (t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("simulation_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()