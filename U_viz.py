#Made on Feb 28 2025
#Base code with the help of chatgpt and preplexity
# But I ahd to go through it to fix some issues and add some features
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def visualize_molecule(atoms, coordinates, filename=-1):
    """
    Visualizes a 3D molecular structure using Matplotlib.

    Parameters:
    atoms (list): List of atomic symbols ('C', 'O', 'H', 'S', 'F', 'Cl', 'I', 'P', 'N').
    coordinates (list of lists): XYZ coordinates of atoms.
    """
    # Atom properties
    atom_colors = {'C': 'black', 'O': 'red', 'H': 'gray', 'S': 'yellow', 'F': 'green', 'Cl': 'lime', 'I': 'purple', 'P': 'orange', 'N': 'blue'}
    atom_sizes = {'C': 500, 'O': 600, 'H': 250, 'S': 700, 'F': 450, 'Cl': 500, 'I': 800, 'P': 650, 'N': 550}

    # Bond properties
    bond_thresholds = {
        ('C', 'H'): 1.09, ('C', 'C'): 1.54, ('C', 'O'): 1.43, ('C', 'N'): 1.47, ('C', 'F'): 1.35, ('C', 'Cl'): 1.76, ('C', 'S'): 1.82,
        ('O', 'H'): 0.96, ('N', 'H'): 1.01, ('S', 'H'): 1.34, ('P', 'H'): 1.44, ('P', 'P'): 2.21, ('S', 'S'): 2.05,
        ('Cl', 'Cl'): 1.99, ('I', 'I'): 2.67,
        ('C', 'C'): 1.34, ('C', 'O'): 1.21, ('C', 'N'): 1.28, ('N', 'N'): 1.24, ('O', 'P'): 1.48,
        ('C', 'C'): 1.20, ('C', 'N'): 1.16, ('N', 'N'): 1.10
    }
    buffer = 0.1  # Buffer region for bond formation

    #max_bonds = {'C': 4, 'O': 2, 'H': 1, 'S': 6, 'F': 1, 'Cl': 1, 'I': 1, 'P': 5, 'N': 3}

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    atoms = np.array(atoms)
    coordinates = np.array(coordinates)

    # Scatter plot for atoms
    for atom, coord in zip(atoms, coordinates):
        ax.scatter(*coord, color=atom_colors[atom], s=atom_sizes[atom], edgecolors='k', label=atom)

    # Compute pairwise distances and create bonds
    num_atoms = len(atoms)
    bonded_atoms = {i: [] for i in range(num_atoms)}

    for i, j in combinations(range(num_atoms), 2):
        atom_pair = tuple(sorted((atoms[i], atoms[j])))
        dist = np.linalg.norm(coordinates[i] - coordinates[j])
        
        if atom_pair in bond_thresholds:
            threshold = bond_thresholds[atom_pair]
            
            if dist <= threshold + buffer:
                bonded_atoms[i].append(j)
                bonded_atoms[j].append(i)
                
                bond_type = 'single'
                if dist <= threshold * 0.9:
                    bond_type = 'double'
                if dist <= threshold * 0.8:
                    bond_type = 'triple'
                if threshold < dist <= threshold + buffer:
                    bond_type = 'partial'
                
                # Draw bonds
                bond_styles = {
                    'single': [0],
                    'double': [-0.05, 0.05],
                    'triple': [-0.1, 0, 0.1],
                    'partial': [0]  # Dashed line for partial bond
                }
                for offset in bond_styles[bond_type]:
                    mid = (coordinates[i] + coordinates[j]) / 2
                    direction = (coordinates[j] - coordinates[i]) / np.linalg.norm(coordinates[j] - coordinates[i])
                    shift = np.cross(direction, [1, 0, 0]) * offset
                    linestyle = '--' if bond_type == 'partial' else '-'
                    ax.plot([coordinates[i, 0] + shift[0], coordinates[j, 0] + shift[0]],
                            [coordinates[i, 1] + shift[1], coordinates[j, 1] + shift[1]],
                            [coordinates[i, 2] + shift[2], coordinates[j, 2] + shift[2]],
                            color='black', linewidth=2, linestyle=linestyle)

    # Labels & display
    ax.set_xlabel('X-axis (Å)')
    ax.set_ylabel('Y-axis (Å)')
    ax.set_zlabel('Z-axis (Å)')
    ax.set_title('3D Molecular Structure')
    
    # Custom legend with uniform marker size
    handles = [plt.Line2D([0.0], [-0.0], marker='o', color='w', markerfacecolor=atom_colors[atom], markersize=8, label=atom) for atom in set(atoms)]
    ax.legend(handles=handles, loc='upper right')
    return ax, plt