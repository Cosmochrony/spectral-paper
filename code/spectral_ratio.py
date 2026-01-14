import numpy as np
import networkx as nx
from scipy.linalg import eigh


def simulate_projective_ratio(n_points=2500):
  # 1. Échantillonnage sur S3 (R4 normalized)
  pts = np.random.normal(size=(n_points, 4))
  pts /= np.linalg.norm(pts, axis=1)[:, np.newaxis]

  # 2. Définition de la structure de fibre (Hopf-like)
  # On définit un champ de vecteurs tangent 'v' qui représente la fibre
  # Sur S3 (x,y,z,w), un vecteur tangent de Hopf est (-y, x, -w, z)
  fibers = np.zeros_like(pts)
  fibers[:, 0] = -pts[:, 1]
  fibers[:, 1] = pts[:, 0]
  fibers[:, 2] = -pts[:, 3]
  fibers[:, 3] = pts[:, 2]

  # 3. Construction de l'adjacence avec anisotropie spectrale
  # On connecte les points proches, mais avec un poids différent
  # si le vecteur de séparation est aligné avec la fibre (cisaillement)
  from scipy.spatial import cKDTree
  tree = cKDTree(pts)
  pairs = tree.query_pairs(r=0.25)

  adj = np.zeros((n_points, n_points))
  for i, j in pairs:
    vec = pts[i] - pts[j]
    vec /= np.linalg.norm(vec)

    # Projection du vecteur de séparation sur la fibre locale
    alignment = np.abs(np.dot(vec, fibers[i]))

    # Cosmochrony logic: le couplage transverse (photon) vs longitudinal (W/Z)
    # On applique la contrainte de "stiffness" projective
    weight = 1.0 + (alignment * (np.sqrt(8 / 3) - 1))
    adj[i, j] = adj[j, i] = weight

  # 4. Calcul du spectre
  L = np.diag(adj.sum(axis=1)) - adj
  evals = eigh(L, eigvals_only=True, subset_by_index=[1, 2])

  l1, l2 = evals[0], evals[1]
  # Dans le régime de saturation de relaxation (BI-like)
  # Le ratio des énergies de mode (carré des fréquences/eigenvalues)
  # tend vers la géométrie de la projection
  observed_ratio = l2 / l1

  print(f"--- Résultats avec Contrainte Projective ---")
  print(f"Mode Transverse (Photon)  l1 : {l1:.4f}")
  print(f"Mode Cisaillement (W/Z)   l2 : {l2:.4f}")
  print(f"Ratio l2/l1                  : {observed_ratio:.4f}")
  print(f"Cible Théorique              : {8 / 3:.4f}")
  print(f"Écart                        : {abs(observed_ratio - 8 / 3) / (8 / 3) * 100:.2f}%")


if __name__ == "__main__":
  simulate_projective_ratio()