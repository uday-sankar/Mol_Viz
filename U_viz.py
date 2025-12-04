import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import chemcoord as cc

def UV(atoms, coordinates, name="Unknown"):
    """
    Visualizes a 3D molecular structure using Matplotlib.

    Parameters:
    atoms (list): List of atomic symbols ('C', 'O', 'H', 'S', 'F', 'Cl', 'I', 'P', 'N').
    coordinates (list of lists): XYZ coordinates of atoms.
    """
    # Atom properties
    atom_colors = {'C': 'black', 'O': 'red', 'H': 'gray', 'S': 'yellow', 'F': 'green', 'Cl': 'lime', 'I': 'purple', 'P': 'orange', 'N': 'blue','Cr':'yellow', 'X':'white'}
    atom_sizes = {'C': 500, 'O': 600, 'H': 250, 'S': 700, 'F': 450, 'Cl': 500, 'I': 800, 'P': 650, 'N': 550, 'Cr':800,'X':400}

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    atoms = np.array(atoms)
    coordinates = np.array(coordinates)

    # Scatter plot for atoms
    for atom, coord in zip(atoms, coordinates):
        ax.scatter(*coord, color=atom_colors[atom], s=atom_sizes[atom], edgecolors='k', label=atom)

    Mol = cc.Cartesian(pd.DataFrame({"atom":atoms, "x":coordinates[:,0], "y":coordinates[:,1], "z":coordinates[:,2]}))
    Bonds = Mol.get_bonds()
    for i in Bonds.keys():
        for j in Bonds[i]:
            direction = (coordinates[j] - coordinates[i]) / np.linalg.norm(coordinates[j] - coordinates[i]) 
            linestyle = '-' 
            ax.plot([coordinates[i, 0] , coordinates[j, 0]],
                    [coordinates[i, 1] , coordinates[j, 1]],
                    [coordinates[i, 2] , coordinates[j, 2]],
                     color='black', linewidth=2, linestyle=linestyle)

    # Labels & display
    ax.set_xlabel('X-axis (Å)')
    ax.set_ylabel('Y-axis (Å)')
    ax.set_zlabel('Z-axis (Å)')
    ax.set_title(name)
    
    # Custom legend with uniform marker size
    #handles = [plt.Line2D([0.0], [-0.0], marker='o', color='w', markerfacecolor=atom_colors[atom], markersize=8, label=atom) for atom in set(atoms)]
    #ax.legend(handles=handles, loc='upper right')
    scaling = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    return ax, plt
