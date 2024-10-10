import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
from scipy.linalg import solve, inv
import re


class mis:
    def __init__(self, rho, mu_d, gan, c_d, sig, beta=1, ponji=1e-9, iter=1500, seed=7):
        """
        Initializes the mis class with specified parameters.

        Args:
            rho (float): Parameter for matrix A.
            mu_d (float): Mean of the endowment.
            gan (float): Gamma parameter.
            c_d (float): Parameter for matrix C.
            sig (float): Sigma parameter for uncertainty.
            beta (float, optional): Discount factor. Defaults to 1.
            iter (int, optional): Number of iterations for state transition. Defaults to 1500.
        """
        # Set constants
        R_kinri = 1 / beta
        ponji_condition = ponji

        # store matrices
        self.Q = np.zeros((3, 3))
        self.Q[2, 2] = ponji_condition
        self.A = np.array(
            [[1.0, 0.0, 0.0], [(1.0 - rho) * mu_d, rho, 0.0], [-gan, 1.0, R_kinri]]
        )
        self.B = np.array([[0.0], [0.0], [1.0]])
        self.C = np.array([[0.0], [c_d], [0.0]])
        self.R = np.array([[1]])

        # Store parameters
        self.gan = gan
        self.iter = iter
        self.mu_d = mu_d
        self.beta = beta
        self.sig = sig
        self.tol = 1e-9
        self.max_iter = 10000
        self.P = None
        self.F = None
        self.no_mis_sig = -1e-9

        # Random number generation
        self.seed = seed
        np.random.seed(self.seed)
        self.eps = np.random.randn(self.iter)

        # Directory creation
        self.make_dir_path = f"./plt_robust_uncertainty/{self.seed}/{beta}_{sig}/{gan}"
        self._create_directory()

    def ABC(self):
        return self.A, self.B, self.C

    def _create_directory(self):
        """Creates a directory if it does not exist."""
        if not os.path.isdir(self.make_dir_path):
            os.makedirs(self.make_dir_path)

    def _olrp(self, beta, A, B, Q, R, W=None):
        """
        Solves the optimal linear regulator problem (OLRP).

        Args:
            beta (float): Discount factor.
            A (numpy.ndarray): State transition matrix.
            B (numpy.ndarray): Control matrix.
            Q (numpy.ndarray): Weight matrix for state.
            R (numpy.ndarray): Weight matrix for control.
            W (numpy.ndarray, optional): Cross-weight matrix. Defaults to None.

        Returns:
            tuple: Tuple containing the feedback matrix and the solution of the Riccati equation.
        """
        # Set tolerance and maximum iterations for convergence
        tol = self.tol
        max_iter = self.max_iter

        # Prepare the cross-weight matrix W if it is not provided
        if W is None:
            W = np.zeros((max(A.shape), B.shape[1]))

        # Initialize the Riccati equation's solution
        p_prev = -0.01 * np.eye(max(A.shape))
        iteration_difference = 1
        iteration_count = 1

        # Iteratively solve the Riccati equation
        while iteration_difference > tol and iteration_count <= max_iter:
            feedback_matrix_prev = solve(
                R + beta * B.T.dot(p_prev).dot(B), beta * B.T.dot(p_prev).dot(A) + W.T
            )
            p_next = (
                beta * A.T.dot(p_prev).dot(A)
                + Q
                - (beta * A.T.dot(p_prev).dot(B) + W).dot(feedback_matrix_prev)
            )
            feedback_matrix_next = solve(
                R + beta * B.T.dot(p_next).dot(B), beta * B.T.dot(p_next).dot(A) + W.T
            )

            # Update the difference and iteration count
            iteration_difference = np.max(
                np.abs(feedback_matrix_next - feedback_matrix_prev)
            )
            iteration_count += 1
            p_prev = p_next

        # Check if the maximum iteration limit was reached
        if iteration_count >= max_iter:
            print("WARNING: Iteration limit reached in OLRP")

        # Return the final feedback matrix and Riccati solution
        feedback_matrix = feedback_matrix_next
        riccati_solution = p_prev

        return feedback_matrix, riccati_solution

    def _calculate_olrprobust(self, sig):
        """
        Calculates the optimal linear regulator problem with robust control.

        This method extends the standard optimal linear regulator problem to
        account for model uncertainty, which is quantified by the parameter 'sig'.

        Args:
            sig (float): Parameter representing the degree of model uncertainty.

        Returns:
            tuple: A tuple containing four elements:
                - F (numpy.ndarray): Feedback matrix for the control law.
                - K (numpy.ndarray): Gain matrix for the uncertainty term.
                - P (numpy.ndarray): Solution to the adjusted Riccati equation.
                - Pt (numpy.ndarray): Adjusted solution to the Riccati equation accounting for uncertainty.

        The matrices F and K are used to adjust the control based on the state
        and the level of uncertainty, respectively. P is the solution to the
        Riccati equation in the standard optimal linear regulator, while Pt is
        the adjusted solution that accounts for model uncertainty.
        """
        Q, R, A, B, C, beta = self.Q, self.R, self.A, self.B, self.C, self.beta
        theta = -1 / sig
        Ba = np.hstack([B, C])

        Ra = np.vstack(
            [
                np.hstack([R, np.zeros((R.shape[0], C.shape[1]))]),
                np.hstack(
                    [
                        np.zeros((C.shape[1], R.shape[0])),
                        -beta * np.eye(C.shape[1]) * theta,
                    ]
                ),
            ]
        )

        f, P = self._olrp(beta, A, Ba, Q, Ra)
        F = f[: B.shape[1], :]
        K = -f[B.shape[1] :, :]
        cTp = np.dot(C.T, P)
        Dp = P + theta ** (-1) * np.dot(
            np.dot(P, C),
            np.dot(inv(np.eye(C.shape[1]) - theta ** (-1) * np.dot(cTp, C)), cTp),
        )

        return F, K, P, Dp

    def olrprobust(self):
        return self._calculate_olrprobust(self.sig)

    def nomis_olrprobust(self):
        return self._calculate_olrprobust(self.no_mis_sig)

    def _calculate_state_transition(self, F, d_first=0, k_first=0):
        """
        Calculates state transition for a given feedback matrix F.

        Args:
            F (numpy.ndarray): Feedback matrix.
            d_first (float, optional): Initial value of endowment. Defaults to 0.
            k_first (float, optional): Initial value of capital. Defaults to 0.

        Returns:
            tuple: Tuple containing the state transition matrix y, capital k, endowment d, and consumption c.
        """
        iter = self.iter
        y = np.zeros((iter + 1, 3))
        y[0, :] = [1.0, d_first, k_first]
        ABF = self.A - self.B @ F

        for i in range(1, iter + 1):
            y[i, :] = ABF @ y[i - 1, :].T + self.C.T * self.eps[i - 1]

        k = y[:, 2]
        d = y[:, 1]
        c = (F @ y.T)[0, :] + self.gan
        return y, k, d, c
    
    def get_Consumption_ave_std(self):
        _, _, _, consumption_mis = self.state_transition()
        return np.average(consumption_mis),np.std(consumption_mis)
    
    def get_Saving_ave_std(self):
        _, k, _, _ = self.state_transition()
        return np.average(k),np.std(k)

    def state_transition(self, d_first=0, k_first=0):
        """
        Calculates the state transition for the misalignment case using the feedback matrix from olrprobust.

        Args:
            d_first (float, optional): Initial endowment. Defaults to 0.
            k_first (float, optional): Initial capital. Defaults to 0.

        Returns:
            tuple: A tuple containing the state matrix (y), capital array (k), endowment array (d), and consumption array (c).
        """
        F = self.olrprobust()[0]
        return self._calculate_state_transition(F, d_first, k_first)

    def nomis_state_transition(self, d_first=0, k_first=0):
        """
        Calculates the state transition for the non-misalignment case using the feedback matrix from nomis_olrprobust.

        Args:
            d_first (float, optional): Initial endowment. Defaults to 0.
            k_first (float, optional): Initial capital. Defaults to 0.

        Returns:
            tuple: A tuple containing the state matrix (y), capital array (k), endowment array (d), and consumption array (c).
        """
        F = self.nomis_olrprobust()[0]
        return self._calculate_state_transition(F, d_first, k_first)

    def mean_consumption(self):
        _, _, _, consumption_mis = self.state_transition()
        _, _, _, consumption_nomis = self.nomis_state_transition()
        print(f"ganmma = {self.gan}")
        print(f"mean of consumption w\o mis = {np.mean(consumption_nomis)}")
        print(f"mean of consumption w\ mis  = {np.mean(consumption_mis)}")

    def _plot_series(
        self, time, data, label, color=None, linestyle=None, linewidth=None
    ):
        """
        Helper function to plot a time series on a Matplotlib plot.

        Args:
            time (array-like): Array of time points.
            data (array-like): Array of data points to plot.
            label (str): Label for the plot series.
            color (str, optional): Color for the plot. Defaults to None.
            linestyle (str, optional): Line style for the plot. Defaults to None.
        """
        plt.plot(
            time,
            data,
            linestyle=linestyle,
            label=label,
            color=color,
            linewidth=linewidth,
        )

    def mis_plot_time_series(self, d_plot=True, y_lim=True, size=True):
        """
        Plots the time series of consumption, endowment, and saving for the misalignment case.
        Saves the plot to a specified directory.
        """
        time_steps = np.arange(self.iter + 1)
        _, savings, endowment, consumption = self.state_transition()

        # plt.title(f"sig = {self.sig}")
        if size == True:
            plt.figure(figsize=[6.5, 6.5])
            if y_lim:
                plt.ylim((-100, 100))
            self._plot_series(
                time_steps, consumption, "consumption", "#000000", "-", linewidth=4
            )
            if d_plot:
                self._plot_series(time_steps, endowment, "endowment", "#000000", "--")
            self._plot_series(
                time_steps[:-1], savings[1:], "saving", "#000000", ":", linewidth=3
            )

            # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
            plt.legend(fontsize=25)
            plt.xlabel("time", fontsize=30)
            # plt.ylabel("comsumption/endowment/saving", fontname="MS Gothic", fontsize=10)
            plt.ylabel("comsumption/saving", fontsize=30)
            plt.tick_params(labelsize=25)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.make_dir_path}/plt_all_mis.png")
        else:
            plt.figure(figsize=[6.5, 4.2])
            if y_lim:
                plt.ylim((-100, 100))
            self._plot_series(time_steps, consumption, "consumption", None, "-")
            if d_plot:
                self._plot_series(time_steps, endowment, "endowment", None, "--")
            self._plot_series(time_steps[:-1], savings[1:], "saving", None, ":")

            # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
            plt.legend(fontsize=20)
            plt.xlabel("time", fontsize=20)
            # plt.ylabel("comsumption/endowment/saving", fontname="MS Gothic", fontsize=10)
            plt.ylabel("comsumption/saving", fontsize=20)
            # plt.tick_params(labelsize=25)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.make_dir_path}/plt_all_mis.png")

    def nomis_plot_time_series(self, d_plot=True, y_lim=True, size=True):
        """
        Plots the time series of consumption, endowment, and saving for the misalignment case.
        Saves the plot to a specified directory.
        """
        time_steps = np.arange(self.iter + 1)
        _, savings, endowment, consumption = self.nomis_state_transition()

        # plt.title(f"sig = 0")
        if size:
            plt.figure(figsize=[6.5, 6.5])
            if y_lim:
                plt.ylim((-100, 100))
            self._plot_series(
                time_steps, consumption, "consumption", "#000000", "-", linewidth=4
            )
            if d_plot:
                self._plot_series(time_steps, endowment, "endowment", "#000000", "--")
            self._plot_series(
                time_steps[:-1], savings[1:], "saving", "#000000", ":", linewidth=3
            )

            # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
            plt.legend(fontsize=25)
            plt.xlabel("time", fontsize=30)
            plt.ylabel("comsumption/saving", fontsize=30)
            plt.tick_params(labelsize=25)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.make_dir_path}/plt_all_nomis.png")
        else:
            plt.figure(figsize=[6.5, 4.2])
            if y_lim:
                plt.ylim((-100, 100))
            self._plot_series(
                time_steps,
                consumption,
                "consumption",
                None,
                "-",
            )
            if d_plot:
                self._plot_series(time_steps, endowment, "endowment", None, "--")
            self._plot_series(time_steps[:-1], savings[1:], "saving", None, ":")

            # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
            plt.legend()
            plt.xlabel("time", fontsize=20)
            plt.ylabel("comsumption/saving", fontsize=20)
            # plt.tick_params(labelsize=25)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.make_dir_path}/plt_all_nomis.png")

    def compare_discounted_utilities(self):
        """
        Compares the discounted utilities for the misaligned and non-misaligned cases.

        Returns:
            tuple: A tuple containing arrays of cumulative discounted utilities and immediate utilities for both cases.
        """
        consumption_mis, consumption_nomis = self.get_consumption_data()
        gamma = self.gan
        beta = self.beta
        time_steps = self.iter

        # Calculate immediate utilities
        immediate_utility_mis = consumption_mis - gamma
        immediate_utility_nomis = consumption_nomis - gamma

        # Calculate cumulative discounted utilities
        discounted_utility_mis = np.cumsum(
            immediate_utility_mis * (beta ** np.arange(time_steps + 1))
        )
        discounted_utility_nomis = np.cumsum(
            immediate_utility_nomis * (beta ** np.arange(time_steps + 1))
        )

        return (
            discounted_utility_nomis,
            discounted_utility_mis,
            immediate_utility_nomis,
            immediate_utility_mis,
        )

    def get_consumption_data(self):
        """
        Retrieves consumption data for both misaligned and non-misaligned cases.

        Returns:
            tuple: A tuple containing arrays of consumption data for the misaligned and non-misaligned cases.
        """
        _, _, _, consumption_mis = self.state_transition()
        _, _, _, consumption_nomis = self.nomis_state_transition()
        return consumption_mis, consumption_nomis

    def plot_immediate_payoff(self):
        """
        Plots the immediate payoff for each period for both misaligned and non-misaligned cases.
        Saves the plot to a specified directory.
        """
        _, _, util_nomis_fun, util_mis_fun = self.compare_discounted_utilities()
        time_steps = np.arange(self.iter + 1)

        plt.figure(figsize=[6.5, 4.2])
        plt.ylim((-20, 20))
        plt.title("Immediate Payoff: (ct - b)")

        # Plot utilities
        self._plot_series(
            time_steps,
            util_nomis_fun,
            f"Non-misalignment (sig={self.no_mis_sig})",
            linestyle="-",
        )
        self._plot_series(
            time_steps, util_mis_fun, f"Misalignment (sig={self.sig})", linestyle="-"
        )

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/immediate_payoff.png")

    def plot_discounted_utility(self):
        """
        Plots the discounted utility over time for both misaligned and non-misaligned cases.
        Saves the plot to a specified directory.
        """
        util_nomis, util_mis, _, _ = self.compare_discounted_utilities()
        time_steps = np.arange(self.iter + 1)

        plt.figure(figsize=[6.5, 4.2])
        plt.ylim((-200, 200))
        plt.title("Discounted Utility Sum")

        # Plot utilities
        self._plot_series(
            time_steps,
            util_nomis,
            f"Non-misalignment (sig={self.no_mis_sig})",
            linestyle="-",
        )
        self._plot_series(
            time_steps, util_mis, f"Misalignment (sig={self.sig})", linestyle="-"
        )

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/Utility.png")
        return plt

    def plot_compare_consumption(self):
        """
        Compares the consumption for misaligned and non-misaligned cases.
        Plots the comparison and saves it to a specified directory.
        """
        _, _, _, consumption_mis = self.state_transition()
        _, _, _, consumption_nomis = self.nomis_state_transition()
        mu_d = self.mu_d
        time_steps = np.arange(self.iter + 1)
        y_range_padding = 100

        plt.figure(figsize=[6.5, 4.2])
        plt.title("Consumption Comparison")
        plt.ylim((mu_d - y_range_padding, mu_d + y_range_padding))

        # Plot consumption

        self._plot_series(
            time_steps, consumption_nomis, f"Non-misalignment (sig={self.no_mis_sig})"
        )
        self._plot_series(time_steps, consumption_mis, f"Misalignment (sig={self.sig})")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/CompareConsumption.png")

    def plot_savings_per_period(self):
        """
        Plots the savings per period for both misaligned and non-misaligned cases.
        Saves the plot to a specified directory.
        """
        _, k_mis, _, consumption_mis = self.state_transition()
        _, k_nomis, _, consumption_nomis = self.nomis_state_transition()
        R_kinri = 1 / self.beta
        time_steps = np.arange(self.iter)

        k_nomis_diff = k_nomis[1:] - R_kinri * k_nomis[:-1]
        k_mis_diff = k_mis[1:] - R_kinri * k_mis[:-1]

        plt.figure(figsize=[6.5, 4.2])
        plt.ylim(-10, 30)
        plt.title("Savings per Period")

        # Plot consumption and savings
        self._plot_series(
            time_steps, consumption_nomis[:-1], f"Consumption (sig={self.no_mis_sig})"
        )
        self._plot_series(
            time_steps, consumption_mis[:-1], f"Consumption (sig={self.sig})"
        )
        self._plot_series(
            time_steps, k_nomis_diff, f"Savings Diff (sig={self.no_mis_sig})"
        )
        self._plot_series(time_steps, k_mis_diff, f"Savings Diff (sig={self.sig})")

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/SavingsPerPeriod.png")

    def plot_savings(self):
        """
        Plots the savings over time for both misaligned and non-misaligned cases.
        Saves the plot to a specified directory.
        """
        _, k_mis, _, _ = self.state_transition()
        _, k_nomis, _, _ = self.nomis_state_transition()
        time_steps = np.arange(self.iter)

        plt.figure(figsize=[6.5, 4.2])
        plt.title("Savings Over Time")

        # Plot savings
        self._plot_series(
            time_steps, k_nomis, f"Nominal Savings (sig={self.no_mis_sig})"
        )
        self._plot_series(time_steps, k_mis, f"Misaligned Savings (sig={self.sig})")

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/PlotSavings.png")

    def plot_consumption_and_savings_comparison(self, d_plot=True, y_lim=True):
        """
        Plots a side-by-side comparison of consumption and savings for both misaligned and non-misaligned cases.
        Saves the plot to a specified directory.

        :param d_plot: If set to 1, plots the endowment along with consumption and savings. Default is 0 (not plotted).
        :param ylim: If set to 0, uses a default y-axis limit based on mean endowment values. Otherwise, the user can specify y-axis limits.
        :return: None. The plot is saved to a file and not returned.
        """
        _, k_mis, d_mis, consumption_mis = self.state_transition()
        _, k_nomis, d_nomis, consumption_nomis = self.nomis_state_transition()
        time_steps = np.arange(self.iter + 1)
        ylim = (self.mu_d - 100, self.mu_d + 100)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[13, 4.2])

        # Plot for Misalignment scenario
        axes[1].set_title(f"sig = {self.sig}", fontsize=20)
        if y_lim:
            axes[1].set_ylim(ylim)
        axes[1].plot(time_steps, consumption_mis, linestyle="-", label="Consumption")
        if d_plot:
            axes[1].plot(time_steps, d_mis, label="endowment")
        axes[1].plot(time_steps[:-1], k_mis[1:], linestyle=":", label="Savings")
        # axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
        axes[1].legend(fontsize=15)
        axes[1].set_xlabel("time", fontsize=20)
        axes[1].set_ylabel("consumption/savings", fontsize=20)
        axes[1].grid()

        # Plot for Non-Misalignment scenario
        axes[0].set_title(f"sig = 0", fontsize=20)
        if y_lim:
            axes[0].set_ylim(ylim)
        axes[0].plot(time_steps, consumption_nomis, linestyle="-", label="Consumption")
        if d_plot:
            axes[0].plot(time_steps, d_nomis, label="endowment")
        axes[0].plot(time_steps[:-1], k_nomis[1:], linestyle=":", label="Savings")
        # axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10})
        axes[0].legend(fontsize=15)
        axes[0].set_xlabel("time", fontsize=20)
        axes[0].set_ylabel("consumption/savings", fontsize=20)
        axes[0].grid()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.make_dir_path}/PlotBoth.png")

    def combine_plots(self, rows, cols):
        title = f"beta={self.beta}/sig={self.sig}/gamma={self.gan}"
        image_paths = [
            f"{self.make_dir_path}/CompareConsumption.png",
            f"{self.make_dir_path}/immediate_payoff.png",
            # f"{self.make_dir_path}/PlotBoth.png",
            f"{self.make_dir_path}/plt_all.png",
            f"{self.make_dir_path}/SavingsPerPeriod.png",
            f"{self.make_dir_path}/Utility.png",
            # ... 他のプロットのパスを追加 ...
        ]

        images = [Image.open(path) for path in image_paths]
        max_width = max(i.size[0] for i in images)
        max_height = max(i.size[1] for i in images)

        total_width = max_width * cols
        total_height = max_height * rows

        # タイトル用のスペースを追加
        title_height = 70  # タイトルの高さ
        total_height += title_height

        new_im = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(new_im)

        # フォントの設定（フォントのパスとサイズを指定）
        font = ImageFont.truetype("arial.ttf", 30)

        # タイトルの描画（テキストサイズを計算せずに中央に描画）
        title_x = (total_width - len(title) * 15) / 2  # 簡易的な中央揃えの計算
        draw.text((title_x, 10), title, font=font, fill=(0, 0, 0))

        x_offset = 0
        y_offset = title_height  # タイトルの高さ分ずらす
        for i, im in enumerate(images):
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]

            if (i + 1) % cols == 0:
                x_offset = 0
                y_offset += im.size[1]

        new_im.save(f"{self.make_dir_path}/combined_image.png")

    def combine_plots(self, rows=3, cols=2):
        self.plot_time_series(d_plot=False, y_lim=True)
        self.compare_discounted_utilities()
        self.plot_immediate_payoff()
        self.plot_discounted_utility()
        self.plot_compare_consumption()
        self.plot_savings_per_period()
        self.plot_consumption_and_savings_comparison()
        # Define the title for the combined image using class parameters
        title = f"beta={self.beta}/sig={self.sig}/gamma={self.gan}"

        # List of paths to the normal-sized plot images
        normal_image_paths = [
            f"{self.make_dir_path}/CompareConsumption.png",
            f"{self.make_dir_path}/immediate_payoff.png",
            # f"{self.make_dir_path}/PlotBoth.png",  # Uncomment if needed
            f"{self.make_dir_path}/plt_all.png",
            f"{self.make_dir_path}/SavingsPerPeriod.png",
            f"{self.make_dir_path}/Utility.png",
            # ... Add other plot paths if necessary ...
        ]

        # Path to the special-sized plot image
        special_image_path = f"{self.make_dir_path}/PlotBoth.png"

        # Combine the normal-sized images
        normal_images = [Image.open(path) for path in normal_image_paths]
        max_width = max(i.size[0] for i in normal_images)
        max_height = max(i.size[1] for i in normal_images)

        # Calculate total dimensions for the combined normal-sized images
        total_width = max_width * cols
        total_normal_height = max_height * rows

        # Create a section for normal-sized images
        normal_section = Image.new(
            "RGB", (total_width, total_normal_height), (255, 255, 255)
        )
        x_offset = 0
        y_offset = 0
        for i, im in enumerate(normal_images):
            normal_section.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]

            # Move to the next row after filling a row
            if (i + 1) % cols == 0:
                x_offset = 0
                y_offset += im.size[1]

        # Load and prepare the special-sized image
        special_image = Image.open(special_image_path)
        title_height = 80  # Increased height for the title space
        total_height = total_normal_height + special_image.size[1] + title_height

        # Create the final combined image
        combined_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.truetype("arial.ttf", 30)

        # Calculate the x-coordinate for centering the title and draw it
        title_x = (total_width - len(title) * 15) / 2
        draw.text((title_x, (title_height - 30) / 2), title, font=font, fill=(0, 0, 0))

        # Place the normal and special-sized sections onto the final image
        combined_image.paste(normal_section, (0, title_height))
        combined_image.paste(special_image, (0, total_normal_height + title_height))

        # Save the final combined image

        combined_image.save(f"{self.make_dir_path}/combined_image.png")
        # 中間画像を削除するためのメソッドを呼び出す
        # all_image_paths = normal_image_paths + [special_image_path]

        # self.cleanup_intermediate_images(all_image_paths)

    def cleanup_intermediate_images(self, image_paths):
        """中間画像を削除する"""
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)

    def create_gif_from_directory(
        self, gif_name="combined_animation.gif", frame_duration=300
    ):
        """
        指定されたディレクトリからGIFを作成する。

        :param base_dir: 画像が含まれている基本ディレクトリのパス。
        :param gif_name: 作成されるGIFファイルの名前。
        :param frame_duration: 各フレームの表示時間（ミリ秒）。
        """
        images = []
        base_dir = f"./plt_robust_uncertainty/{self.seed}/{self.beta}_{self.sig}"
        # 基本ディレクトリ内のすべてのサブディレクトリを取得
        subdirectories = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        # 数字であるサブディレクトリ名のみを選択
        digit_subdirs = [d for d in subdirectories if re.match(r"^\d+$", d)]

        # 数字の順序に従ってソート
        digit_subdirs.sort(key=int)

        # 各サブディレクトリから画像を読み込む
        for subdir in digit_subdirs:
            image_path = os.path.join(base_dir, subdir, "combined_image.png")
            if os.path.exists(image_path):
                images.append(Image.open(image_path))

        # 画像が見つかった場合、GIFを作成
        if images:
            gif_path = os.path.join(base_dir, gif_name)
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                loop=0,
                duration=frame_duration,
            )
            print(f"GIFが保存されました: {gif_path}")
        else:
            print("指定されたパスに画像が見つかりませんでした。")
