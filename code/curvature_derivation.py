import numpy as np


def derive_geometric_ratio(n_samples=100000):
  """
  Simule la densité de rigidité spectrale sur une 3-sphère fibrée.
  On compare l'énergie de déformation transverse (Base)
  à l'énergie de déformation longitudinale (Fibre).
  """
  # 1. Échantillonnage de vecteurs de séparation unitaires dans R4
  # représentant les directions de relaxation possibles
  vecs = np.random.normal(size=(n_samples, 4))
  vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]

  # 2. Définition de l'opérateur de projection II (Fibré de Hopf)
  # Sur S3, la direction de la fibre en un point (x,y,z,w) est (-y, x, -w, z)
  # On simule ici l'alignement statistique moyen des modes de relaxation

  # On définit une direction de fibre arbitraire pour le calcul local
  # (Par symétrie, le résultat est indépendant du point choisi sur S3)
  fiber_direction = np.array([1, 0, 0, 0])

  # 3. Calcul de la rigidité spectrale (Stiffness)
  # Mode l1 : Relaxation transverse à la fibre (Base / Photon)
  # Mode l2 : Relaxation alignée avec la fibre (Cisaillement / Bosons)

  projections = np.abs(np.dot(vecs, fiber_direction))

  # Énergie de déformation de la base (modes transverses)
  # C'est la moyenne des projections sur le complémentaire de la fibre
  stiffness_base = np.mean(1 - projections ** 2)

  # Énergie de déformation de la fibre (modes de torsion)
  # Dans le régime de saturation, cela correspond à la densité de flux
  # de relaxation contraint par la courbure de Hopf.
  stiffness_fiber = np.mean(3 * projections ** 2)

  # Le ratio de courbure spectrale
  # R = (Densité de Rigidité Fibre) / (Densité de Rigidité Base)
  # Dans la limite continue du substrat :
  ratio = (stiffness_fiber / stiffness_base)

  print(f"--- Dérivation de la Courbure Spectrale ---")
  print(f"Rigidité Base (Transverse)  : {stiffness_base:.4f}")
  print(f"Rigidité Fibre (Torsion)    : {stiffness_fiber:.4f}")
  print(f"-------------------------------------------")
  print(f"Ratio Émergent l2/l1        : {ratio:.4f}")
  print(f"Cible Théorique (8/3)       : {8 / 3:.4f}")
  print(f"Erreur Relative             : {abs(ratio - 8 / 3) / (8 / 3) * 100:.4f}%")
  print(f"-------------------------------------------")


if __name__ == "__main__":
  derive_geometric_ratio()