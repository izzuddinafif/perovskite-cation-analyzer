from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_average_bond_lengths(structure, a_cation_list, b_cation_list, anion_list):
    cnn = CrystalNN()
    a_bonds = []
    b_bonds = []

    for site in structure.sites:
        element = site.specie.symbol
        neighbors = cnn.get_nn_info(structure, structure.index(site))
        for neighbor in neighbors:
            neighbor_elem = neighbor['site'].specie.symbol
            dist = neighbor['site'].distance(site)
            if element in a_cation_list and neighbor_elem in anion_list:
                a_bonds.append(dist)
            elif element in b_cation_list and neighbor_elem in anion_list:
                b_bonds.append(dist)

    return np.mean(a_bonds), np.mean(b_bonds)

def calculate_tolerance_from_structure(cif_path, a_cations, b_cations, anions=["O"]):
    structure = Structure.from_file(cif_path)
    r_ax, r_bx = get_average_bond_lengths(structure, a_cations, b_cations, anions)
    t = r_ax / (np.sqrt(2) * r_bx)
    return t

# === USER SETTINGS ===
cif_folder = "."  # Folder containing CIF files
a_cations = ["Cs", "FA"]
b_cations = ["Pb"]
anions = ["I"]
output_csv = "tolerance_factors.csv"

results = []

# === MAIN LOOP ===
for filename in os.listdir(cif_folder):
    if filename.endswith(".cif"):
        cif_path = os.path.join(cif_folder, filename)
        try:
            t = calculate_tolerance_from_structure(cif_path, a_cations, b_cations, anions)
            results.append({"Filename": filename, "ToleranceFactor": t})
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# === EXPORT TO CSV ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")

# === PLOT ===
if not df.empty:
    plt.figure(figsize=(10, 5))
    plt.bar(df["Filename"], df["ToleranceFactor"], color="skyblue")
    plt.axhline(1, color="red", linestyle="--", label="Ideal tolerance factor = 1")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Tolerance Factor")
    plt.title("Perovskite Tolerance Factors")
    plt.tight_layout()
    plt.legend()
    plt.savefig("tolerance_factors_plot.png")
    plt.show()
