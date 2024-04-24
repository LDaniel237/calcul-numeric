import numpy as np
from numpy.linalg import cholesky, norm, svd, inv

def jacobi_method(A, tol=1e-9, max_iterations=100):
    """
    Computes the eigenvalues and eigenvectors of a symmetric matrix A using the Jacobi method.
    The function returns the eigenvalues and the corresponding eigenvectors.
    """
    # Ensure A is a numpy array.
    A = np.array(A)
    n = A.shape[0]
    U = np.eye(n)
    
    # Function to find the indices of the pivot element
    def find_pivot(A):
        n = A.shape[0]
        max_val = 0
        k = 0
        l = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if abs(A[i,j]) >= max_val:
                    max_val = abs(A[i,j])
                    k = i
                    l = j
        return k, l, max_val
    
    # Main loop of the Jacobi method
    for iteration in range(max_iterations):
        k, l, max_val = find_pivot(A)
        
        if max_val < tol:
            # Convergence achieved
            break
        
        # Calculate the Jacobi rotation matrix
        if A[k, k] == A[l, l]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[k, l] / (A[k, k] - A[l, l]))
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Apply the Jacobi rotation
        J = np.eye(n)
        J[k, k] = c
        J[l, l] = c
        J[k, l] = s
        J[l, k] = -s
        
        # Update the matrices A and U
        A = np.matmul(np.matmul(J.T, A), J)
        U = np.matmul(U, J)
    
    # The eigenvalues are on the diagonal of A and the columns of U are the eigenvectors
    eigenvalues = np.diagonal(A)
    eigenvectors = U
    
    return eigenvalues, eigenvectors

# Test the function on a symmetric matrix example
A_example = np.array([[2, 1],
                      [1, 2]])

eigenvalues, eigenvectors = jacobi_method(A_example)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

####################################################################################################

def compute_cholesky_series(A, k_max, epsilon):
    """
    Compute the series of matrices A^(k) using Cholesky decomposition.

    Parameters:
    A (np.array): Initial positive-definite symmetric matrix.
    k_max (int): Maximum number of iterations.
    epsilon (float): Tolerance for the stopping criterion.

    Returns:
    A^(k) (np.array): The last matrix computed in the series.
    k (int): The number of iterations performed.
    """

    # Function to check if a matrix is positive definite
    def is_positive_definite(B):
        try:
            _ = cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    # Ensure the initial matrix is positive-definite
    if not is_positive_definite(A):
        raise ValueError("The initial matrix is not positive-definite.")

    k = 0
    while k < k_max:
        # Compute the Cholesky decomposition
        L = cholesky(A)

        # Calculate the next matrix in the series
        A_next = np.matmul(L.T, L);

        # Check the stopping criterion
        if norm(A_next - A) < epsilon:
            break

        # Update the matrix for the next iteration
        A = A_next
        k += 1

    return A, k

# Test the function with a random positive-definite symmetric matrix
n = 3  # Dimension of the matrix
A_initial = np.random.rand(n, n)
A_initial = np.matmul(A_initial, A_initial.T)  # Ensuring the matrix is symmetric and positive-definite
print("Initial matrix:")
print(A_initial)

k_max = 100  # Maximum number of iterations
epsilon = 1e-6  # Tolerance

# Calculate the series
A_final, num_iterations = compute_cholesky_series(A_initial, k_max, epsilon)
print("Number of iterations:", num_iterations)
print("Final matrix:")
print(A_final)

####################################################################################################

def svd_properties(A):
    # Compute the Singular Value Decomposition
    U, s, Vt = svd(A)
    m, n = A.shape

    # Compute the rank of the matrix as the number of non-zero singular values
    rank = np.sum(s > 1e-10)

    # Compute the number of conditioning of the matrix A
    if rank > 0:
        conditioning = s[0] / s[rank-1]
    else:
        conditioning = np.inf

    # Compute the Moore-Penrose pseudoinverse
    S_inv = np.zeros_like(A.T, dtype=float)
    np.fill_diagonal(S_inv, 1/s[s > 1e-10])
    A_pseudo = np.matmul(Vt.T, np.matmul(S_inv, U.T))

    # Compute the least squares pseudoinverse
    A_lsq_pseudo = np.matmul(inv(np.matmul(A.T, A)), A.T)

    # Compute the norm of the difference between the two pseudoinverses
    norm_difference = norm(A_pseudo - A_lsq_pseudo)

    return s, rank, conditioning, A_pseudo, A_lsq_pseudo, norm_difference

# Define a matrix with more rows than columns (p > n)
p, n = 4, 3
A = np.random.randn(p, n)

# Calculate the properties using SVD
singular_values, matrix_rank, cond_number, A_pseudo, A_lsq_pseudo, norm_diff = svd_properties(A)

print("Singular values:", singular_values)
print("Matrix rank:", matrix_rank)
print("Condition number:", cond_number)
print("Moore-Penrose pseudoinverse:")
print(A_pseudo)
print("Least squares pseudoinverse:")
print(A_lsq_pseudo)
print("Norm of the difference between the two pseudoinverses:", norm_diff)