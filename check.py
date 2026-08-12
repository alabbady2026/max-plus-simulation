import numpy as np

INF = -np.inf

def maxplus_mult(A, B):
    """Max-Plus Matrix Multiplication: C_ij = max_k (A_ik + B_kj)"""
    m, n1 = A.shape
    n2, p = B.shape
    assert n1 == n2, "Matrix dimensions do not match"
    C = np.full((m, p), INF)
    for i in range(m):
        for j in range(p):
            C[i, j] = np.max(A[i, :] + B[:, j])
    return C

def maxplus_add(A, B):
    """Max-Plus Matrix Addition: C_ij = max(A_ij, B_ij)"""
    return np.maximum(A, B)

def maxplus_kleene_star(A0):
    """Computes (A0)* = I ⊕ A0 ⊕ A0^2 ⊕ ... until convergence"""
    n = A0.shape[0]
    I = np.full((n, n), INF)
    np.fill_diagonal(I, 0)
    
    A_star = I.copy()
    A_pow = I.copy()
    for _ in range(n):
        A_pow = maxplus_mult(A_pow, A0)
        A_star = maxplus_add(A_star, A_pow)
    return A_star

def format_maxplus(M):
    """Helper function to cleanly display ε for -inf"""
    return np.where(np.isneginf(M), "ε", M)

A0 = np.array([
    [INF, INF, INF, INF],
    [INF, INF, INF, INF],
    [  2,   1, INF, INF],
    [INF, INF,   5, INF]
])

A1 = np.array([
    [  3, INF, INF, INF],
    [INF,   4, INF, INF],
    [INF, INF, INF,   3],
    [INF, INF, INF, INF]
])

B = np.array([
    [  0, INF],
    [INF,   0],
    [INF, INF],
    [INF, INF]
])

C0 = np.array([
    [INF, INF, INF, 2]
])


A0_star = maxplus_kleene_star(A0)

A = maxplus_mult(A0_star, A1)

B0 = maxplus_mult(A0_star, B)

C = maxplus_mult(C0, A0_star)

print("--- Canonical State Matrix A = (A0)* ⊗ A1 ---")
print(format_maxplus(A))

print("\n--- Canonical Input Matrix B0 = (A0)* ⊗ B ---")
print(format_maxplus(B0))

print("\n--- Canonical Output Matrix C = C0 ⊗ (A0)* ---")
print(format_maxplus(C))