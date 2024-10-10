import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
from numpy import dot, eye
from scipy.linalg import solve, inv


class no_mis:
    def __init__(self, rho, mu_d, gan, c_d, beta=1,ponji=1e-9, iter=1500,seed=7):
        R_kinri = 1 / beta
        
        Q = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, ponji]])
        A = np.array(
            [[1.0, 0.0, 0.0], [(1.0 - rho) * mu_d, rho, 0.0], [-gan, 1.0, R_kinri]]
        )
        # u=\gamma - c_tにしたため符号を反転した
        B = np.array([[0.0], [0.0], [1.0]])
        C = np.array([[0.0], [c_d], [0.0]])
        R = np.array([[1]])
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.gan = gan
        self.iter = iter
        self.k, self.n = self.Q.shape[0], self.R.shape[0]
        if C is None:
            # == If C not given, then model is deterministic. Set C=0. == #
            self.j = 1
            self.C = np.zeros((self.n, self.j))
        else:
            self.C = C

        self.beta = beta
        self.sig = -1e-9
        self.tol = 1e-9
        self.max_iter = 1000
        self.P = None
        self.F = None

        np.random.seed(seed)
        self.eps = np.random.randn(self.iter)

        # 作成するディレクトリのpathを指定
        make_dir_path = "./plt_no_uncertainty"

        # 作成しようとしているディレクトリが存在するかどうかを判定する
        if os.path.isdir(make_dir_path):
            # 既にディレクトリが存在する場合は何もしない
            pass
        else:
            # ディレクトリが存在しない場合のみ作成する
            os.makedirs(make_dir_path)

    def olrp(self, beta, A, B, Q, R, W=None):
        tol, max_iter = (
            self.tol,
            self.max_iter,
        )
        m = max(A.shape)
        rb, cb = B.shape

        if W is None:
            W = np.zeros((m, cb))
        p0 = -0.01 * np.eye(m)
        dd = 1
        it = 1
        while dd > tol and it <= max_iter:
            f0 = solve(R + beta * B.T.dot(p0).dot(B), beta * B.T.dot(p0).dot(A) + W.T)
            p1 = beta * A.T.dot(p0).dot(A) + Q - (beta * A.T.dot(p0).dot(B) + W).dot(f0)
            f1 = solve(R + beta * B.T.dot(p1).dot(B), beta * B.T.dot(p1).dot(A) + W.T)
            dd = np.max(np.abs(f1 - f0))
            it += 1
            p0 = p1
        if it >= max_iter:
            print("WARNING: Iteration limit of 1000 reached in OLRP")

        f = f1
        p = p0

        return f, p

    def olrprobust(self):
        Q, R, A, B, C, sig, beta = (
            self.Q,
            self.R,
            self.A,
            self.B,
            self.C,
            self.sig,
            self.beta,
        )
        theta = -1 / sig
        Ba = np.hstack([B, C])
        rR, cR = R.shape
        rC, cC = C.shape
        Ra = np.vstack(
            [
                np.hstack([R, np.zeros((rR, cC))]),
                np.hstack([np.zeros((cC, cR)), -beta * np.eye(cC) * theta]),
            ]
        )
        f, P = self.olrp(beta, A, Ba, Q, Ra)
        rB, cB = B.shape
        F = f[:cB, :]
        rf, cf = f.shape
        K = -f[cB:rf, :]
        cTp = dot(C.T, P)
        Pt = P + theta ** (-1) * dot(
            dot(P, C), dot(inv(eye(cC) - theta ** (-1) * dot(cTp, C)), cTp)
        )
        return F, K, P, Pt

    def state_transition(self, d_first=0, k_first=0):
        F = self.olrprobust()[0]
        iter = self.iter
        y = np.array([[1.0, d_first, k_first]])
        ABF = self.A - self.B @ F
        for i in range(iter):
            y = np.insert(y, 0, ABF @ y[0].T + self.C.T * self.eps[i], axis=0)

        y = y[::-1]
        k = y[:, 2]
        d = y[:, 1]
        c = np.zeros(iter + 1)
        for i in range(iter + 1):
            c[i] = (F @ y[i])[0] + self.gan
        return y, k, d, c

    def plt_c_k_d(self,d_plot=True,y_lim=True):
        plt.figure(figsize=[6.5, 4.2])
        iter_t = self.iter
        y, k, d, c = self.state_transition()
        plt.title("sig = %f" % (self.sig))
        colorlist = ["#FF4B00", "#005AFF", "#03AF7A"]
        if y_lim:
            plt.ylim((-100, 100))
        plt.plot(
            np.array(range(iter_t + 1)),
            c[: iter_t + 1],
            linestyle="-",
            label="consumption",
            color=colorlist[0],
        )
        if d_plot:
            plt.plot(
                np.array(range(iter_t + 1)),
                d[: iter_t + 1],
                linestyle="--",
                label="endowment",
                color=colorlist[1],
            )
        plt.plot(
            np.array(range(iter_t)),
            k[1 : iter_t + 1],
            linestyle=":",
            label="saving",
            color=colorlist[2],
        )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, prop={"size": 10}
        )
        plt.xlabel("time", fontsize=10)
        plt.ylabel("comsumption/endowment/saving", fontname="MS Gothic", fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig("./plt_no_uncertainty/plt_all.png")
