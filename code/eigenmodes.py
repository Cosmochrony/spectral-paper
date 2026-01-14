import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# =============================================
# PARAMÈTRES PHYSIQUES (recalibrés pour cibler m₁~0.5 MeV)
# =============================================
K_0 = 1e30       # Couplage ajusté [s⁻²]
a = 1e-15        # Espacement du réseau [m]
chi_c = 1e-35    # Échelle de χ [m]
L = 64           # Taille du réseau
target_masses = [0.5, 100, 1000]  # Masses cibles en MeV

# =============================================
# 1. CONSTRUCTION DU LAPLACIEN AVEC 3 SOLITONS
# =============================================
def build_laplacian_3d(L, K_0, chi_c):
    N = L**3
    laplacian = lil_matrix((N, N))

    # Positions des 3 solitons (arrangement tétraédrique)
    soliton_positions = [
        (L//4, L//4, L//4),      # Soliton 1
        (3*L//4, 3*L//4, L//4),   # Soliton 2
        (L//4, 3*L//4, 3*L//4),   # Soliton 3
        (3*L//4, L//4, 3*L//4)    # Soliton 4 (pour stabilité)
    ]

    for i in range(L):
        for j in range(L):
            for k in range(L):
                idx = i + j*L + k*L**2
                neighbors = [
                    ((i+1)%L, j, k), ((i-1)%L, j, k),
                    (i, (j+1)%L, k), (i, (j-1)%L, k),
                    (i, j, (k+1)%L), (i, j, (k-1)%L)
                ]

                # Couplage non-uniforme (dépend de la distance aux solitons)
                coupling = K_0
                for (si, sj, sk) in soliton_positions:
                    distance = np.sqrt((i-si)**2 + (j-sj)**2 + (k-sk)**2)
                    if distance < L//8:  # Proche d'un soliton
                        coupling *= 10.0  # Renforce le couplage

                laplacian[idx, idx] = 6 * coupling
                for (ni, nj, nk) in neighbors:
                    n_idx = ni + nj*L + nk*L**2
                    laplacian[idx, n_idx] = -coupling
    return laplacian.tocsr()

# =============================================
# 2. CALCUL DES VALEURS PROPRES (filtre les modes triviaux)
# =============================================
def compute_eigenvalues(laplacian, n_eigenvalues=5):
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigenvalues, which='SM')
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e20]  # Seuil pour éliminer les modes nuls
    return eigenvalues[0:3], eigenvectors[:, 0:3]

# =============================================
# 3. CONVERSION EN MASSES (avec χ_c et hbar_eff)
# =============================================
def eigenvalues_to_masses(eigenvalues, K_0, a, chi_c):
    c = 2.99792458e8      # m/s
    # hbar_eff = χ_c² * K_0 / c (échelle d'action effective)
    hbar_eff = chi_c**2 * K_0 / c
    # Facteur de conversion : λ_n [s⁻²] → m_n [MeV]
    # m_n = √(λ_n) * (hbar_eff/c²) * (1/a) * 1e6 (pour MeV)
    conversion_factor = np.sqrt(hbar_eff / (c**2 * 1.602176634e-19)) * (1 / a) * 1e6 * 1e-18
    masses_mev = np.sqrt(eigenvalues) * conversion_factor
    return masses_mev

# =============================================
# 4. VISUALISATION (mode propre non-trivial)
# =============================================
def plot_eigenmode(eigenvector, L):
    mode_3d = np.reshape(eigenvector, (L, L, L))
    mid = L // 2
    plt.figure(figsize=(10, 6))
    plt.imshow(mode_3d[:, :, mid], cmap='viridis', vmin=-0.8, vmax=0.8)
    plt.colorbar(label='Amplitude')
    plt.title(r'First Non-Trivial Eigenmode of $\Delta_G^{(0)}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# =============================================
# EXÉCUTION PRINCIPALE
# =============================================
if __name__ == "__main__":
    print("=== Cosmochrony Simulation: 3-Soliton Eigenvalues of Δ_G^(0) ===")
    print(f"Parameters: L={L}, K_0={K_0:.1e} s⁻², a={a:.1e} m, χ_c={chi_c:.1e} m")

    # 1. Construction du Laplacien avec 3 solitons
    laplacian = build_laplacian_3d(L, K_0, chi_c)

    # 2. Calcul des valeurs propres
    eigenvalues, eigenvectors = compute_eigenvalues(laplacian)
    print(f"Filtered eigenvalues (s⁻²): {eigenvalues}")

    # 3. Conversion en masses (en MeV)
    masses_mev = eigenvalues_to_masses(eigenvalues, K_0, a, chi_c)
    print(f"Computed masses (MeV): {masses_mev}")
    print(f"Target masses (MeV):   {target_masses}")

    # 4. Visualisation du premier mode non-trivial
    if len(eigenvalues) > 0:
        plot_eigenmode(eigenvectors[:, 0], L)

    # 5. Comparaison avec les cibles
    print("\n=== RESULTS ===")
    for n, (computed, target) in enumerate(zip(masses_mev, target_masses)):
        error = abs(computed - target) / target * 100
        print(f"Mode {n+1}: Computed = {computed:.2f} MeV | Target = {target:.1f} MeV | Error = {error:.1f}%")
