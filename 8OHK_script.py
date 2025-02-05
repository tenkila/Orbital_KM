import numpy as np

# Define basic matrices
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
identity = np.eye(2, dtype=complex)

def generate_fermionic_operators(N):
    """
    Generate annihilation and creation operators for N orbitals with spin.
    
    Parameters:
    N (int): Number of orbitals.
    
    Returns:
    tuple: (annihilation_ops, creation_ops) where each is a list of 2N matrices.
           The order is [orbital_0↑, orbital_0↓, orbital_1↑, orbital_1↓, ...].
    """
    total_modes = 2 * N  # Each orbital has two modes: spin-up and spin-down
    annihilation_ops = []
    creation_ops = []
    
    for p in range(total_modes):
        # Construct annihilation operator for mode p
        op = 1  # Start with scalar 1 to build Kronecker product
        for q in range(total_modes):
            if q < p:
                op = np.kron(op, sigma_z)
            elif q == p:
                op = np.kron(op, sigma_minus)
            else:
                op = np.kron(op, identity)
        annihilation_ops.append(op)
        # Creation operator is the conjugate transpose
        creation_ops.append(op.conj().T)
    
    return annihilation_ops, creation_ops

# Example usage:
N = 8  # Number of orbitals
cs, cdags = generate_fermionic_operators(N)
print(cs[0])  # Annihilation operator for orbital 0, spin up
