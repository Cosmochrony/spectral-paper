import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class CosmoChronyNetwork:
  """
  Simulation du substrat χ sur un réseau relationnel.
  Calcule le Laplacien spectral ∆^(0)_G et extrait les valeurs propres.
  """

  def __init__(self, N=100, topology='cubic_3d', chi_c=1.0, K0=1.0):
    """
    Args:
        N: Nombre de nœuds (pour topologie 1D/2D) ou N^3 (pour 3D)
        topology: 'chain', 'grid_2d', 'cubic_3d', 'random_geometric'
        chi_c: Échelle de corrélation caractéristique
        K0: Rigidité maximale de relaxation
    """
    self.topology = topology
    self.chi_c = chi_c
    self.K0 = K0

    # Construction du réseau
    if topology == 'chain':
      self.nodes = N
      self.positions = np.arange(N).reshape(-1, 1)
    elif topology == 'grid_2d':
      n = int(np.sqrt(N))
      self.nodes = n * n
      x, y = np.meshgrid(np.arange(n), np.arange(n))
      self.positions = np.column_stack([x.ravel(), y.ravel()])
    elif topology == 'cubic_3d':
      n = int(np.cbrt(N))
      self.nodes = n ** 3
      x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
      self.positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    elif topology == 'random_geometric':
      self.nodes = N
      self.positions = np.random.rand(N, 3)  # Points aléatoires dans [0,1]^3

    # Initialisation du champ χ
    self.chi = np.random.randn(self.nodes) * 0.2  # Petites fluctuations initiales

    # Construction de la matrice de couplage K_ij
    self.K_matrix = self._build_coupling_matrix()

    # Construction du Laplacien ∆^(0)_G
    self.laplacian = self._build_laplacian()

  def _build_coupling_matrix(self):
    """
    Construit K_ij selon Eq. 12 du document :
    K_ij = K0 * exp(-(Δχ_ij)^2 / χ_c^2)

    Pour l'instant, on utilise χ̄ (champ lissé) pour éviter la circularité.
    """
    # Distance combinatoriale (topologique)
    d_comb = squareform(pdist(self.positions, metric='euclidean'))

    # Champ de fond χ̄ (moyenne locale pour éviter circularité)
    chi_bar = self._compute_background_field()

    # Différences de χ̄ entre nœuds
    delta_chi = np.abs(chi_bar[:, None] - chi_bar[None, :])

    # Couplage constitutif (Eq. 12)
    K = self.K0 * np.exp(-(delta_chi ** 2) / (self.chi_c ** 2))

    # Connexions uniquement entre voisins proches (rayon de coupure)
    cutoff = self._get_cutoff_radius()
    K[d_comb > cutoff] = 0

    return csr_matrix(K)

  def _compute_background_field(self):
    """
    Calcule χ̄_i par moyenne relationnelle (Appendice F.5)
    """
    # Distance combinatoriale pour définir voisinages
    d_comb = squareform(pdist(self.positions, metric='euclidean'))
    cutoff = self._get_cutoff_radius()

    chi_bar = np.zeros(self.nodes)
    for i in range(self.nodes):
      neighbors = d_comb[i] <= cutoff
      chi_bar[i] = np.mean(self.chi[neighbors])

    return chi_bar

  def _get_cutoff_radius(self):
    """Rayon de coupure pour connectivité locale"""
    if self.topology in ['chain', 'grid_2d', 'cubic_3d']:
      return 1.5  # Plus proches voisins + seconds voisins
    else:
      return 0.2  # Pour réseaux aléatoires

  def _build_laplacian(self):
    """
    Construit le Laplacien de graphe pondéré :
    ∆^(0)_G = D - K
    où D_ii = Σ_j K_ij
    """
    # Matrice de degré
    degree = np.array(self.K_matrix.sum(axis=1)).flatten()
    D = diags(degree, 0, format='csr')

    # Laplacien
    L = D - self.K_matrix

    return L

  def compute_spectrum(self, n_eigenvalues=20):
    """
    Calcule les n premières valeurs propres de ∆^(0)_G

    Returns:
        eigenvalues: Valeurs propres λ_n (ordonnées)
        eigenvectors: Vecteurs propres correspondants
    """
    # IMPORTANT : eigsh calcule les plus petites valeurs propres
    # qui=SM signifie "smallest magnitude"
    eigenvalues, eigenvectors = eigsh(self.laplacian, k=n_eigenvalues, which='SM')

    # Tri par ordre croissant
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

  def analyze_spectral_ratios(self, eigenvalues):
    """
    Analyse les ratios λ_n/λ_1 et cherche λ_2/λ_1 ≈ 8/3
    """
    # Exclure la première valeur propre (λ_0 ≈ 0 pour graphe connecté)
    lambda_1 = eigenvalues[1]  # Premier mode non-trivial

    ratios = eigenvalues[1:] / lambda_1

    results = {
      'lambda_1': lambda_1,
      'lambda_2/lambda_1': ratios[1] if len(ratios) > 1 else None,
      'target_ratio': 8 / 3,
      'deviation': abs(ratios[1] - 8 / 3) if len(ratios) > 1 else None,
      'all_ratios': ratios
    }

    return results

  def add_topological_constraint(self, winding_number=1):
    """
    Ajoute une contrainte topologique simulant un soliton de nombre d'enroulement w.
    Modifie localement le champ χ pour créer une excitation topologique.

    Args:
        winding_number: w = 1 (electron), w = 2 (muon analogue), w = 3 (proton)
    """
    center = self.nodes // 2

    if self.topology == 'cubic_3d':
      # Créer un vortex topologique au centre
      n = int(np.cbrt(self.nodes))
      center_pos = self.positions[center]

      for i in range(self.nodes):
        r_vec = self.positions[i] - center_pos
        r = np.linalg.norm(r_vec)

        if r > 0:
          # Phase azimutale (comme un vortex)
          theta = np.arctan2(r_vec[1], r_vec[0])

          # Configuration de type skyrmion/vortex
          # Pour w=1 : χ ∝ exp(iθ) → partie réelle
          # Pour w>1 : χ ∝ exp(iwθ)
          profile = np.exp(-r ** 2 / (2 * self.chi_c ** 2))
          self.chi[i] += winding_number * profile * np.cos(winding_number * theta)

    # Reconstruire K et ∆^(0)_G avec la nouvelle configuration
    self.K_matrix = self._build_coupling_matrix()
    self.laplacian = self._build_laplacian()


def plot_spectrum_analysis(network, eigenvalues, eigenvectors, results):
  """
  Visualisation complète du spectre et des modes propres
  """
  fig, axes = plt.subplots(2, 3, figsize=(18, 12))

  # 1. Spectre de ∆^(0)_G
  ax = axes[0, 0]
  ax.plot(eigenvalues, 'o-', linewidth=2, markersize=8)
  ax.axhline(y=results['lambda_1'], color='r', linestyle='--',
             label=f"$\\lambda_1$ = {results['lambda_1']:.3e}")
  if results['lambda_2/lambda_1'] is not None:
    ax.axhline(y=results['lambda_1'] * 8 / 3, color='g', linestyle='--',
               label=f'λ₁ × 8/3 = {results["lambda_1"] * 8 / 3:.3e}')
  ax.set_xlabel('Mode index n', fontsize=12)
  ax.set_ylabel(r"Eigenvalue $\lambda_n$", fontsize=12)
  ax.set_title(r"Spectrum of $\Delta^{(0)}_G$", fontsize=14, fontweight="bold")
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 2. Ratios λ_n/λ_1
  ax = axes[0, 1]
  ratios = results['all_ratios']
  ax.plot(range(len(ratios)), ratios, 'o-', linewidth=2, markersize=8)
  ax.axhline(y=8 / 3, color='r', linestyle='--', linewidth=2,
             label=f'Target: 8/3 = {8 / 3:.3f}')
  if results['lambda_2/lambda_1'] is not None:
    ax.plot(1, results['lambda_2/lambda_1'], 'ro', markersize=15,
            label=f'λ₂/λ₁ = {results["lambda_2/lambda_1"]:.3f}')
  ax.set_xlabel('Mode index n', fontsize=12)
  ax.set_ylabel('λₙ / λ₁', fontsize=12)
  ax.set_title('Spectral Ratios', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 3. Écart au ratio cible
  ax = axes[0, 2]
  deviations = np.abs(ratios - 8 / 3)
  ax.semilogy(range(len(deviations)), deviations, 'o-', linewidth=2)
  ax.axhline(y=0.1, color='g', linestyle='--', label='10% tolerance')
  ax.set_xlabel('Mode index n', fontsize=12)
  ax.set_ylabel('|λₙ/λ₁ - 8/3|', fontsize=12)
  ax.set_title('Deviation from Target Ratio', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 4-6. Visualisation des 3 premiers modes propres non-triviaux
  for idx, mode_idx in enumerate([1, 2, 3]):
    ax = axes[1, idx]

    if network.topology == 'cubic_3d':
      # Projection sur un plan z = z_mid
      n = int(np.cbrt(network.nodes))
      z_mid = n // 2

      mode = eigenvectors[:, mode_idx].real
      mode_3d = mode.reshape(n, n, n)
      mode_slice = mode_3d[:, :, z_mid]

      im = ax.imshow(mode_slice, cmap='RdBu_r', aspect='auto')
      ax.set_title(f'Eigenmode ψ_{mode_idx} (λ={eigenvalues[mode_idx]:.3e})',
                   fontsize=12, fontweight='bold')
      plt.colorbar(im, ax=ax)

    elif network.topology == 'grid_2d':
      n = int(np.sqrt(network.nodes))
      mode = eigenvectors[:, mode_idx].real
      mode_2d = mode.reshape(n, n)

      im = ax.imshow(mode_2d, cmap='RdBu_r', aspect='auto')
      ax.set_title(f'Eigenmode ψ_{mode_idx} (λ={eigenvalues[mode_idx]:.3e})',
                   fontsize=12, fontweight='bold')
      plt.colorbar(im, ax=ax)

    else:  # chain
      mode = eigenvectors[:, mode_idx].real
      ax.plot(mode, linewidth=2)
      ax.set_title(f'Eigenmode ψ_{mode_idx} (λ={eigenvalues[mode_idx]:.3e})',
                   fontsize=12, fontweight='bold')
      ax.grid(True, alpha=0.3)

  plt.tight_layout()
  return fig


def run_comprehensive_analysis():
  """
  Lance une analyse complète sur différentes topologies et tailles de réseau
  """
  print("=" * 80)
  print(" COSMOCHRONY SPECTRAL ANALYSIS: Search for λ₂/λ₁ = 8/3 ".center(80))
  print("=" * 80)
  print()

  topologies = ['chain', 'grid_2d', 'cubic_3d']
  sizes = {
    'chain': [100, 500, 1000],
    'grid_2d': [100, 400, 900],  # √N × √N
    'cubic_3d': [125, 512, 1000]  # ∛N × ∛N × ∛N
  }

  all_results = {}

  for topo in topologies:
    print(f"\n{'=' * 80}")
    print(f" Topology: {topo.upper()} ".center(80))
    print('=' * 80)

    all_results[topo] = []

    for N in sizes[topo]:
      print(f"\n  → Network size: {N} nodes")

      # Construction du réseau
      network = CosmoChronyNetwork(N=N, topology=topo, chi_c=1.0, K0=1.0)

      # Calcul du spectre
      eigenvalues, eigenvectors = network.compute_spectrum(n_eigenvalues=20)

      # Analyse des ratios
      results = network.analyze_spectral_ratios(eigenvalues)

      # Stockage
      all_results[topo].append({
        'N': N,
        'eigenvalues': eigenvalues,
        'results': results
      })

      # Affichage
      print(f"     λ₁ = {results['lambda_1']:.6e}")
      if results['lambda_2/lambda_1'] is not None:
        print(f"     λ₂/λ₁ = {results['lambda_2/lambda_1']:.6f}")
        print(f"     Target: 8/3 = {8 / 3:.6f}")
        print(f"     Deviation: {results['deviation']:.6f} ({results['deviation'] * 100:.2f}%)")

        # Verdict
        if results['deviation'] < 0.1:
          print(f"     ✅ EXCELLENT MATCH (< 10% error)")
        elif results['deviation'] < 0.5:
          print(f"     ✓  Good match (< 50% error)")
        else:
          print(f"     ✗  Poor match (> 50% error)")

    # Visualisation pour la plus grande taille
    N_max = sizes[topo][-1]
    network_final = CosmoChronyNetwork(N=N_max, topology=topo, chi_c=1.0, K0=1.0)
    eigenvalues, eigenvectors = network_final.compute_spectrum(n_eigenvalues=20)
    results_final = network_final.analyze_spectral_ratios(eigenvalues)

    fig = plot_spectrum_analysis(network_final, eigenvalues, eigenvectors, results_final)
    fig.suptitle(f'Spectral Analysis: {topo.upper()} (N={N_max})',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.savefig(f'cosmochrony_spectrum_{topo}.png', dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: cosmochrony_spectrum_{topo}.png")

  # Résumé comparatif
  print(f"\n{'=' * 80}")
  print(" COMPARATIVE SUMMARY ".center(80))
  print('=' * 80)

  summary_data = []
  for topo in topologies:
    for entry in all_results[topo]:
      summary_data.append({
        'Topology': topo,
        'N': entry['N'],
        'λ₂/λ₁': entry['results']['lambda_2/lambda_1'],
        'Deviation': entry['results']['deviation']
      })

  print("\n{:<15} {:<10} {:<15} {:<15}".format('Topology', 'N', 'λ₂/λ₁', 'Deviation'))
  print('-' * 60)
  for row in summary_data:
    if row['λ₂/λ₁'] is not None:
      print("{:<15} {:<10} {:<15.6f} {:<15.6f}".format(
        row['Topology'], row['N'], row['λ₂/λ₁'], row['Deviation']))

  return all_results


def test_with_topological_excitations():
  """
  Test avec excitations topologiques (solitons de différents nombres d'enroulement)
  """
  print("\n" + "=" * 80)
  print(" TESTING WITH TOPOLOGICAL EXCITATIONS ".center(80))
  print("=" * 80)

  winding_numbers = [1, 2, 3]  # Électron, muon-analogue, proton-analogue

  for w in winding_numbers:
    print(f"\n  → Winding number w = {w}")

    network = CosmoChronyNetwork(N=512, topology='cubic_3d', chi_c=1.0, K0=1.0)
    network.add_topological_constraint(winding_number=w)

    eigenvalues, eigenvectors = network.compute_spectrum(n_eigenvalues=20)
    results = network.analyze_spectral_ratios(eigenvalues)

    print(f"     λ₁ = {results['lambda_1']:.6e}")
    if results['lambda_2/lambda_1'] is not None:
      print(f"     λ₂/λ₁ = {results['lambda_2/lambda_1']:.6f}")
      print(f"     Deviation from 8/3: {results['deviation']:.6f}")


if __name__ == "__main__":
  # Analyse complète
  results = run_comprehensive_analysis()

  # Test avec excitations topologiques
  test_with_topological_excitations()

  print("\n" + "=" * 80)
  print(" ANALYSIS COMPLETE ".center(80))
  print("=" * 80)
  print("\nInterpretation:")
  print("  • If λ₂/λ₁ ≈ 8/3 emerges naturally → Strong evidence for Cosmochrony")
  print("  • If λ₂/λ₁ varies significantly with topology → Fine-tuning required")
  print("  • If λ₂/λ₁ ~ 8/3 only with topological constraints → Mechanism validated")
  print()