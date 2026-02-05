import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chemcoord as cc
import py3Dmol as py3D

class MoleculeVisualizer:
    """
    Class to visualize molecules using py3Dmol, animate normal modes, and
    animate paths (e.g., MEP).
    """

    atom_colors = {
        'C': 'black', 'O': 'red', 'H': 'gray', 'S': 'yellow',
        'F': 'green', 'Cl': 'lime', 'I': 'purple', 'P': 'orange',
        'N': 'blue', 'Cr': 'yellow', 'X': 'white'
    }

    atom_sizes = {
        'C': 500, 'O': 600, 'H': 250, 'S': 700, 'F': 450,
        'Cl': 500, 'I': 800, 'P': 650, 'N': 550,
        'Cr': 800, 'X': 400
    }

    def __init__(self, atoms, coords, int_to_cart=None, cart_to_int=None):
        """
        Docstring for __init__
        Necessary Params:
            * self: molcular vizualization code
            * atoms: an list of all the atom symbols. eg ["H", "H"] for H$_2$
            * coords: a numpy array of coordinates (Cartesian or internal)
        Optional params:
            * int_to_cart: Internal to cartesian coordinate conversion function
            * cart_to_int: Cartesian to internal conversion function
            Both these functions should work nested. 
            We should be able to write: int_to_cart(cart_to_int(int_to_cart))
        """
        self.atoms = np.array(atoms)
        self.coords = np.array(coords)
        self.int_to_cart = int_to_cart
        self.cart_to_int = cart_to_int
        shape = coords.shape
        if len(shape) > 1:
            if shape[1] == 3:
                print("Cartesian Coordinates Deteceted")
                self.coord_type = "Cartesian"
            else:
                self.coord_type = "N/A"
        else:
            print(" Flattened internal Coordinates Deteceted ")
            self.coord_type = "Internal"
        if int_to_cart == None:
            print("No conversion between internal to cartesian")
        else:
            print("Internal to cartesian transformation provided")
        if cart_to_int == None:
            print("No conversion between cartesian to internal")
        else:
            print("Cartesian to internal transformation provided")
        if (int_to_cart !=None and cart_to_int != None):
            self.Consistancy_check( coords, tol=1e-4)


    def _get_bonds(self, coords=None):
        """Compute bonds using chemcoord."""
        if coords is None:
            coords = self.coords
        mol = cc.Cartesian(
            pd.DataFrame({"atom": self.atoms,
                          "x": coords[:, 0],
                          "y": coords[:, 1],
                          "z": coords[:, 2]})
        )
        return mol.get_bonds()

    def _coords_to_xyz_string(self, coords, atoms=None, comment=""):
        """Convert coordinates to XYZ format string."""
        if atoms is None:
            atoms = self.atoms
        Num_atoms = len(atoms)
        try:
            coords_xyz = coords.reshape(Num_atoms, 3)
        except:
            if self.int_to_cart !=-1:
                coords_xyz = coords.flatten()
                coords_xyz = self.int_to_cart(coords)
            else:
                print("!!! Issue with coordinates supplied !!!\n !!! Can't respahe to Nx3 and no internal to cartesian conversion supplied\n")
                exit()

        
        xyz_string = f"{len(atoms)}\n{comment}\n"
        for atom, coord in zip(atoms, coords_xyz):
            xyz_string += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
        return xyz_string

    def visualize(self, coords=None, name="Unknown", 
                  bond_radius=0.1, sphere_radius=0.3, 
                  bond_color='white', bond_opacity=0.9,
                  width=500, height=500):
        """
        Draw a 3D molecular structure with bonds using py3Dmol.
        
        Parameters
        ----------
        coords : np.ndarray, optional
            Coordinates to visualize. If None, uses self.coords
        name : str
            Title/comment for the structure
        bond_radius : float
            Thickness of bonds
        sphere_radius : float
            Size of atom spheres
        bond_color : str
            Color of bonds
        bond_opacity : float
            Opacity of bonds (0-1)
        width : int
            Viewer width in pixels
        height : int
            Viewer height in pixels
            
        Returns
        -------
        view : py3Dmol.view
            The py3Dmol viewer object
        """
        if coords is None:
            coords = self.coords
        
        xyz_string = self._coords_to_xyz_string(coords, comment=name)
        
        view = py3D.view(width=width, height=height)
        view.addModel(xyz_string, 'xyz')
        view.setStyle({
            'stick': {
                'radius': bond_radius,
                'color': bond_color,
                'opacity': bond_opacity
            },
            'sphere': {'radius': sphere_radius}
        })
        view.zoomTo()
        return view

    def visualize_matplotlib(self, coords=None, name="Unknown", ax=-1):
        """
        Draw a 3D molecular structure with bonds using matplotlib.
        (Original matplotlib implementation kept for compatibility)
        """
        if coords is None:
            coords = self.coords
        
        Num_atoms = len(self.atoms)
        try:
            coords_xyz = coords.reshape(Num_atoms, 3)
        except:
            if self.int_to_cart !=-1:
                coords_xyz = coords.flatten()
                coords_xyz = self.int_to_cart(coords)
            else:
                print("!!! Issue with coordinates supplied !!!\n !!! Can't respahe to Nx3 and no internal to cartesian conversion supplied\n")
                exit()
        if ax == -1:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")

        # Scatter atoms
        for atom, coord in zip(self.atoms, coords_xyz):
            ax.scatter(*coord,
                       color=self.atom_colors[atom],
                       s=self.atom_sizes[atom],
                       edgecolors="k")

        # Bonds
        bonds = self._get_bonds(coords_xyz)
        for i in bonds.keys():
            for j in bonds[i]:
                ax.plot([coords_xyz[i, 0], coords_xyz[j, 0]],
                        [coords_xyz[i, 1], coords_xyz[j, 1]],
                        [coords_xyz[i, 2], coords_xyz[j, 2]],
                        color="black", linewidth=2)

        # Labels
        ax.set_xlabel("X-axis (Å)")
        ax.set_ylabel("Y-axis (Å)")
        ax.set_zlabel("Z-axis (Å)")
        ax.set_title(name)

        # Scale equal
        scaling = np.array([getattr(ax, f'get_{dim}lim')() for dim in "xyz"])
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)

        return ax, plt

    def animate_normal_mode(self, mode_vector, scale=0.2, n_frames=30,
                            interval=100, save_file="normal_mode.gif",title="Normal Mode Animation"):
        """
        Animate a normal mode vibration using matplotlib.
        """
        coords = self.coords
        t_vals = np.linspace(0, 2 * np.pi, n_frames)
        displacements = [coords + scale * np.sin(t) * mode_vector for t in t_vals]

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        scatters = [ax.scatter([], [], [],
                               color=self.atom_colors[a],
                               s=self.atom_sizes[a],
                               edgecolors="k") for a in self.atoms]

        bonds = self._get_bonds(coords)
        bond_lines = []
        for i in bonds.keys():
            for j in bonds[i]:
                line, = ax.plot([], [], [], color="black", linewidth=2)
                bond_lines.append((line, i, j))

        # Limits
        scaling = np.array([coords.min(), coords.max()])
        ax.set_xlim(scaling.min() - 1, scaling.max() + 1)
        ax.set_ylim(scaling.min() - 1, scaling.max() + 1)
        ax.set_zlim(scaling.min() - 1, scaling.max() + 1)
        ax.set_title(title)

        def update(frame_idx):
            frame_coords = displacements[frame_idx]
            for scatter, coord in zip(scatters, frame_coords):
                scatter._offsets3d = ([coord[0]], [coord[1]], [coord[2]])
            for line, i, j in bond_lines:
                line.set_data([frame_coords[i, 0], frame_coords[j, 0]],
                              [frame_coords[i, 1], frame_coords[j, 1]])
                line.set_3d_properties([frame_coords[i, 2], frame_coords[j, 2]])
            return scatters + [l for l, _, _ in bond_lines]

        ani = animation.FuncAnimation(fig, update,
                                      frames=n_frames, interval=interval)
        if save_file.endswith(".gif"):
            ani.save(save_file, writer="pillow")
        else:
            ani.save(save_file, writer="ffmpeg")
        plt.close(fig)
        print(f"Animation saved to {save_file}")

    def visualize_mep_trajectory_py3dmol(self, beads, ports, width=600, height=600):
        """
        Visualize MEP trajectory as multi-model structure in py3Dmol.
        Use slider to navigate through frames.
        
        Parameters
        ----------
        beads : 2D array-like
            beads[i,j] is a dict with keys "Energy" and "geom"
        ports : list of tuple
            List of bead indices (i,j) to connect
        width : int
            Viewer width
        height : int
            Viewer height
            
        Returns
        -------
        view : py3Dmol.view
            Interactive viewer with trajectory
        """
        # Build path
        full_path = self._build_mep_path(beads, ports)
        
        # Convert to geometries
        if self.int_to_cart == -1:
            coords_path = [beads[i, j]["geom"] for (i, j) in full_path]
        else:
            coords_path = [self.int_to_cart(beads[i, j]["geom"]) for (i, j) in full_path]
        
        # Create multi-model XYZ string
        xyz_multi = ""
        for idx, geom in enumerate(coords_path):
            energy = beads[full_path[idx]]["Energy"]
            xyz_multi += self._coords_to_xyz_string(geom, comment=f"Frame {idx}, E={energy:.4f}")
        
        # Visualize with py3Dmol
        view = py3D.view(width=width, height=height)
        view.addModelsAsFrames(xyz_multi, 'xyz')
        view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'radius': 0.3}})
        view.animate({'loop': 'forward'})
        view.zoomTo()
        return view

    def _build_mep_path(self, beads, ports):
        """Helper to build MEP path through beads."""
        def trace_guided_mep(start, end):
            path = [start]
            i0, j0 = start
            i1, j1 = end
            n_steps = max(abs(i1 - i0), abs(j1 - j0)) + 1
            straight_line = [
                (int(round(i0 + (i1 - i0) * t / n_steps)),
                 int(round(j0 + (j1 - j0) * t / n_steps)))
                for t in range(1, n_steps + 1)
            ]
            for target in straight_line:
                ti, tj = target
                candidates = [
                    (ii, jj)
                    for ii in range(ti - 2, ti + 3)
                    for jj in range(tj - 2, tj + 3)
                    if 0 <= ii < len(beads) and 0 <= jj < len(beads[0])
                ]
                next_point = min(candidates, key=lambda x: beads[x]["Energy"])
                if next_point != path[-1]:
                    path.append(next_point)
            return path

        full_path = []
        for k in range(len(ports) - 1):
            seg_path = trace_guided_mep(ports[k], ports[k + 1])
            if k > 0:
                seg_path = seg_path[1:]
            full_path.extend(seg_path)
        return full_path

    def animate_mep_from_beads(self, beads, ports, filename="MEP_test.mp4", interval=10):
        """
        Animate molecular geometries along MEP using matplotlib.
        (Original matplotlib implementation)
        """
        full_path = self._build_mep_path(beads, ports)

        # Convert path to geometries
        if self.int_to_cart == -1:
            coords_path = [beads[i, j]["geom"] for (i, j) in full_path]
        else:
            coords_path = [self.int_to_cart(beads[i, j]["geom"]) for (i, j) in full_path]

        # Precompute global axis limits
        all_coords = np.vstack(coords_path)
        x_min, y_min, z_min = all_coords.min(axis=0) - 1
        x_max, y_max, z_max = all_coords.max(axis=0) + 1

        # Animation setup
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.cla()
            geom = coords_path[frame]
            energy = beads[full_path[frame]]["Energy"]
            self.visualize_matplotlib(coords=geom, ax=ax, name=f"Frame {frame}, E={energy:.4f}")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            return ax,

        ani = animation.FuncAnimation(fig, update,
                                      frames=len(coords_path),
                                      interval=interval, blit=False)

        ani.save(filename, writer="ffmpeg", fps=5)
        print(f"Animation saved to {filename}")

    def save_mep_as_xyz(self, beads, ports, filename="MEP_trajectory.xyz"):
        """
        Save the MEP along beads as an XYZ trajectory file.
        """
        full_path = self._build_mep_path(beads, ports)

        # Convert path to geometries
        if self.int_to_cart == -1:
            coords_path = [beads[i, j]["geom"] for (i, j) in full_path]
        else:
            coords_path = [self.int_to_cart(beads[i, j]["geom"]) for (i, j) in full_path]

        # Write XYZ trajectory file
        with open(filename, "w") as f:
            for idx, geom in enumerate(coords_path):
                f.write(f"{len(self.atoms)}\n")
                energy = beads[full_path[idx]]["Energy"]
                f.write(f"Frame {idx}, Energy = {energy:.6f} kcal/mol\n")
                for atom, coord in zip(self.atoms, geom):
                    f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        print(f"MEP trajectory saved to {filename}")

    def animate_mode_internal(
        self,
        mode_vector_int,
        q0_int = None,
        filename="mode_animation",
        n_frames=30,
        amplitude=0.1,
        filetype="xyz"
    ):
        """
        Animate a normal mode in internal coordinates and save as .xyz or .molden file.
        """
        atoms = self.atoms
        if q0_int == None:
            q0_int = self.coords
        ##
        int_to_cart = self.int_to_cart
        #cart_to_int = self.cart_to_int
        if int_to_cart == None:
            print("Animation not possible without internal to cartesian conversion function")
        assert filetype.lower() in ["xyz", "molden"], "Filetype must be 'xyz' or 'molden'"
        mode_norm = mode_vector_int / np.linalg.norm(mode_vector_int)
        frames = []
        print(f"Generating {n_frames} frames for normal mode animation in internal coordinates...")
        print(f"Amplitude: {amplitude:.4f}, File type: {filetype}")

        for i in range(n_frames):
            phase = 2 * np.pi * i / n_frames
            q_disp = q0_int + amplitude * np.sin(phase) * mode_norm
            R_cart = int_to_cart(q_disp)
            frames.append(R_cart)
            print(f"Frame {i + 1:3d}/{n_frames}: displacement phase = {phase:.3f} rad")

        if filetype.lower() == "xyz":
            outfile = f"{filename}.xyz"
            with open(outfile, "w") as f:
                for R in frames:
                    f.write(f"{len(atoms)}\n\n")
                    for sym, (x, y, z) in zip(atoms, R):
                        f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
            print(f"XYZ animation saved to {outfile}")

        elif filetype.lower() == "molden":
            outfile = f"{filename}.molden"
            with open(outfile, "w") as f:
                f.write("[Molden Format]\n[GEOMETRIES] XYZ\n")
                for idx, R in enumerate(frames):
                    f.write(f"Geom_{idx+1}\n")
                    for sym, (x, y, z) in zip(atoms, R):
                        f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
            print(f"MOLDEN animation saved to {outfile}")

        print("Normal mode animation generation complete.\n")

    def save_normal_mode_xyz(
        self,
        mode_vector,
        mode_egval = None,
        coords = None,
        filename="mode_animation",
        n_frames=30,
        amplitude=0.1,
        filetype="xyz",
        verbose = 0
    ):
        """
        save the animation file for a normal mode motion in xyz
        """
        atoms = self.atoms
        int_to_cart = self.int_to_cart
        ##
        if coords == None:
            coords = self.coords
            if self.coord_type != "Cartesian":
                if int_to_cart == None:
                    print("Animation not possible. No method to convert from internal to cartesian")
                else:
                    coords = int_to_cart(coords)
        xyz_shape = coords.shape
        #cart_to_int = self.cart_to_int
        if int_to_cart == None:
            print("Animation not possible without internal to cartesian conversion function")
        assert filetype.lower() in ["xyz", "molden"], "Filetype must be 'xyz' or 'molden'"
        mode_norm = mode_vector.flatten() / np.linalg.norm(mode_vector.flatten())
        frames = []
        if verbose > 0:
            print(f"Generating {n_frames} frames for normal mode animation in internal coordinates...")
            print(f"Amplitude: {amplitude:.4f}, File type: {filetype}")

        for i in range(n_frames):
            phase = 2 * np.pi * i / n_frames
            dxyz_flat = coords.flatten() + amplitude * np.sin(phase) * mode_norm
            #R_cart = int_to_cart(q_disp)
            frames.append(dxyz_flat.reshape(xyz_shape))
            if verbose > 0:
                print(f"Frame {i + 1:3d}/{n_frames}: displacement phase = {phase:.3f} rad")

        if filetype.lower() == "xyz":
            outfile = f"{filename}.xyz"
            with open(outfile, "w") as f:
                for R in frames:
                    f.write(f"{len(atoms)}\n\n")
                    for sym, (x, y, z) in zip(atoms, R):
                        f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
                if mode_egval != None:
                    f.write(f"Eigen Value of Mode: {mode_egval}")

        elif filetype.lower() == "molden":
            outfile = f"{filename}.molden"
            with open(outfile, "w") as f:
                f.write("[Molden Format]\n[GEOMETRIES] XYZ\n")
                for idx, R in enumerate(frames):
                    f.write(f"Geom_{idx+1}\n")
                    for sym, (x, y, z) in zip(atoms, R):
                        f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
                if mode_egval != None:
                    f.write(f"\nEigen Value of Mode: {mode_egval}")

        print(f"Normal mode animation generation complete. File saved as {outfile}")

    def read_xyz_Traj(self, filename):
        """
        Reads an XYZ trajectory file and returns a list of frames.
        Each frame is a list of lines (strings).
        """
        frames = []
        with open(filename, 'r') as f:
            while True:
                # Read atom count line
                n_atoms = f.readline()
                if not n_atoms:
                    break  # end of file

                comment = f.readline()  # energy/frame line
                frame = [n_atoms, comment]

                # Read the next N atom lines
                for _ in range(int(n_atoms.strip())):
                    frame.append(f.readline())

                frames.append(frame)

        return frames
    
    def Consistancy_check(self, test_geom,tol=1e-4):
        # All functions should work with numpy matrix/array. test_geom has to be numpy array or matrix
        # Interanal coordinates should be flattened arrays, cartesian coordinates should ne Nx3 matrix
        cart_to_int = self.cart_to_int
        int_to_cart = self.int_to_cart
        ##
        N = np.linalg.norm
        single_tr = False
        double_tr = False
        geom_shape = test_geom.shape
        if len(geom_shape) > 1:
            if geom_shape[1] == 3:
                print("_________ Input geometry Cartesian _________\n\t Attempting Single transformration (Cart) \n\t Cart0 -> Int0 -> Cartn; |Cartn - Cart0|")
                g0 = np.copy(test_geom)
                int0 = cart_to_int(g0)
                g0_back = int_to_cart(int0)
                dG = N(g0_back - g0)
                if  dG <= tol:
                    print("\tSingle Transformation Check Sucessfull (dG<tol).\n\t |dG| =",dG,"\n\t _____________________")
                    single_tr = True
                else:
                    print("!!!!!!!!! Single Transformation not withing tolerance !!!!!!!!!\n\t |dG| =",dG,"\n\t _____________________")
                    single_tr =  False
                print("\t Attempting Double Transformation (Internal)\n\t Int0 -> Cartn -> Intn; |int0 - Intn| ")
                int0_back = cart_to_int(g0_back)
                dG_back = N(int0_back - int0)
                if  dG_back <= tol:
                    print("\tDouble Transformation Sucessfull. \n\t No issues in coordinate transformation back and forth\n\t|dG| =",dG_back)
                    double_tr =  True
                else:
                    print("!!!!!!!!! Double Transformation not withing tolerance !!!!!!!!!\n |dG| =",dG_back,"\n _______________________________")
                    double_tr = False
            else:
                print("!!!!!!!!! Unknown Geometry Input !!!!!!!!!")
                double_tr,  single_tr =  False, False
        elif len(geom_shape)  == 1:
            print("_________ Input geometry Internal _________\n\t Attempting Single transformration (Int) \n\t Int0 -> Cart0 -> Intn; |Intn - Int0|")
            int0 = np.copy(test_geom) 
            g0 = int_to_cart(int0)
            int0_back = cart_to_int(g0)
            dG = N(int0_back - int0)
            if  dG <= tol:
                print("\t Single Transformation Check  Sucessfull. \n\t No issues in coordinate transformation back and forth\n\t |dG| =",dG,"\n\t _____________________")
                single_tr =  True
            else:
                print("!!!!!!!!! Transformation not withing tolerance !!!!!!!!!\n |dG| =",dG,"\n\t _____________________")
                single_tr =  False
            print("\tAttempting Double Transformation (Internal)\n\t Cart0 -> Intn -> Cartn; |Cart0 - Cartn| ")
            g0_back = int_to_cart(int0_back)
            dG_back =  N(g0_back-g0)
            if  dG_back <= tol:
                print("\tDouble Transformation Sucessfull. \n\t No issues in coordinate transformation back and forth\n\t |dG| =",dG_back)
                double_tr =  True
            else:
                print("!!!!!!!!! Double Transformation not withing tolerance !!!!!!!!!\n |dG| =",dG_back,"\n _______________________________")
                double_tr = False
        else:
            print(" !!!!! Unknown Geometry Input !!!!!!!")
            double_tr,  single_tr =  False, False
        if (single_tr == True and double_tr == True):
            print("Both Single and Double Transformations Sucess")
        elif (single_tr == False and double_tr == True):
            print("Single transformation failure starting from a cartesian geometry is expected.\nThis happens because the internal coordinates loose translational and rotational information.\nThe true consistance can be checked by Calling the function using an internal coordinate,\nwhich will give both transfermations correctly only if the conversions are true. ")    
        return single_tr, double_tr