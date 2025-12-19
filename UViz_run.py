from U_viz import visualize_molecule as UV
atoms = ['C', 'O', 'H', 'H', 'N']
coordinates = [[0.0,0.0,0.],[1.21, 0, 0], [-0.55, 0.96, 0], [-0.55, -0.96, 0], [2.0, 0, 0]]
ax, plt = UV(atoms, coordinates)
plt.savefig("Test.png",dpi=450)