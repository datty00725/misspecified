import numpy as np
from numpy import dot, eye
from math import sqrt
from scipy.linalg import solve, eig, norm, inv


def asdf():
    print("4")


def doubleo(A, C, Q, R, tol=1e-15):
    a0 = A.T
    b0 = C.T.dot(solve(R, C))
    # print(f"b0 ={b0}")
    g0 = Q
    dd = 1.0
    ss = max(A.shape)
    v = np.eye(ss)
    c_vec = C.shape[0] > 1
    # print(f"c_vec = {c_vec}")
    while dd > tol:
        a1 = a0.dot(solve(v + b0.dot(g0), a0))
        # print("a1.shape %d %d"%(a1.shape[0],a1.shape[1]))

        b1 = b0 + a0.dot(solve(v + b0.dot(g0), b0.dot(a0)))
        # print("b1.shape %d %d"%(b1.shape[0],b1.shape[1]))
        # print(f"b1 ={b1}")
        g1 = g0 + a0.T.dot(g0).dot(solve(v + b0.dot(g0), a0))
        # print("g1.shape %d %d"%(g1.shape[0],g1.shape[1]))
        # print("A.dot(g1).dot(C.T).shape %d %d"%(A.dot(g1).dot(C.T).shape[0],A.dot(g1).dot(C.T).shape[1]))
        # print("(C.dot(g1).dot(C.T) + R).shape %d %d"%((C.dot(g1).dot(C.T) + R).shape[0],(C.dot(g1).dot(C.T) + R).shape[1]))
        if c_vec:
            # print("in if")
            k1 = np.dot(A.dot(g1), solve(np.dot(C, g1.T).dot(C.T) + R.T, C).T)
            k0 = np.dot(A.dot(g0), solve(np.dot(C, g0.T).dot(C.T) + R.T, C).T)
        else:
            k1 = np.dot(np.dot(A, g1), C.T / (np.dot(C, g1).dot(C.T) + R))
            k0 = np.dot(A.dot(g0), C.T / (np.dot(C, g0).dot(C.T) + R))
        dd = np.max(np.abs(k1 - k0))
        a0 = a1
        b0 = b1
        g0 = g1

    K = k1
    S = g1
    # print(f"K ={K}")
    # print(f"S ={S}")

    return K, S


# f, P = olrp(beta, A, Ba, Q, Ra)が渡される
def olrp(beta, A, B, Q, R, W=None, tol=1e-7, max_iter=1000):
    m = max(A.shape)
    rb, cb = B.shape

    if W is None:
        W = np.zeros((m, cb))

    if type(R) != np.ndarray:
        R = np.array([[R]])

    print(f"eig={eig(R)}")
    
        # Rの固有値が1e-5より大きいかチェック
        # このIf分岐の意味が分からない。
        #
        # scipy.linalg.eig() は行列の固有値と固有ベクトルを返します。
        # 戻り値のデータ型は常に complex (複素数型) です。
        #

        # print("in np.min(np.abs(eig(R)[0])) > 1e-5")
    A = sqrt(beta) * (A - B.dot(solve(R, W.T)))
    B = sqrt(beta) * B
    Q = Q - W.dot(solve(R, W.T))
    k, s = doubleo(A.T, B.T, Q, R)
    # print(f"k= {k}")
    # print(f"s= {s}")
    f = k.T + solve(R, W.T)
    p = s
    # print(p)
   
    return f, p


def olrprobust(beta, A, B, C, Q, R, sig):
    """
    目的関数(the robust control problem)　-x'Px　=　min sum beta^t(x'Qx + u'R u)

    遷移関数　x' = A x + B u + C w

    行動　u_t = - F x_t


    引数
    ==========
    beta : float
        これは割引率
    A, B, C : 行列, dtype = float
        遷移関数の係数
    Q, R : 行列, dtype = float
        the robust control problemの係数
        Rは3×3(ホント？)、Q=1との記述あり(2.8.3節に記述あり)
    sig :
        意思決定者の頑健性の考慮具合
        sig < 0 の時、頑健性を意識している
        sig = -1 / thetaで与えられる

    返り値
    =======
    F : 行列, dtype = float
        u_t = - F x_t
        のF
    P : 行列, dtype = float
        正定値行列である必要がある。
        -x'PxのP
    Pt : 行列, dtype = float
        D(P)のことをこれで表すとする。
        2.8.5節で出てくる。
        -y' D(P) y = -y' Pt yであり、yはxの次の値
    K : 行列, dtype = float
        以下で与えられる最悪の場合を表すやつ
        w_{t+1} = K x_t


    備考
    =====
    目的関数は最小問題であるからR,Qも正定値である必要がある。"""
    # print("in orlprobust")
    # ここは定義通り
    theta = -1 / sig

    # 2.7.1a式のこと
    Ba = np.hstack([B, C])
    rR, cR = R.shape
    rC, cC = C.shape

    #  there is only one shock
    # 2.7.1b式のこと
    Ra = np.vstack(
        [
            np.hstack([R, np.zeros((rR, cC))]),
            np.hstack([np.zeros((cC, cR)), -beta * np.eye(cC) * theta]),
        ]
    )
    # print(f"sig = {sig}")
    # print("R shape =%d %d"%R.shape)
    # print("Ra shape =%d %d"%Ra.shape)
    # print(f"in olrprobust -beta * np.eye(cC) * theta = {-beta * np.eye(cC) * theta}")
    # print(f"in olrprobust Ra = {Ra}")
    # orlpは最適線形規制問題（Optimal Linear Regulation Problem）の略
    # 別のファイルで定義されている。
    # fは何者なんだろう
    f, P = olrp(beta, A, Ba, Q, Ra)

    rB, cB = B.shape

    # スライスで抜き出しているが、何をしているのか
    F = f[:cB, :]
    rf, cf = f.shape

    # fという行列からFとKが取れた。
    # このことは文章中に記述なし。
    # 但し2.5章と関係がありそう。
    K = -f[cB:rf, :]

    # C'*Pと毎回打つの面倒なので
    cTp = dot(C.T, P)

    # 2.5.6式のこと
    Pt = P + theta ** (-1) * dot(
        dot(P, C), dot(inv(eye(cC) - theta ** (-1) * dot(cTp, C)), cTp)
    )
    return F, K, P, Pt
