#####################
### Hückel Solver ###
#####################

'''
Author: Goran Giudetti
Date:   04/07/2023
Description: Calculates eigenfunctions and related properties of the Huckel hamiltonian 
for a given system. 
Inputs:
    - xyz file of molecule (checked for planarity and connectivity)
    - Connectivity matrix (optional - override default)
    - Hückel Hamiltonian (optional - override default)
    - Number of electrons (optional - override default)
'''

import numpy as np # linear algebra library
import scipy.linalg as la
from scipy.linalg import issymmetric
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# argparse support for boolean values
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to load XYZ coordinates from a file
def load_xyz(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0].strip())
        atoms = []
        coordinates = []
        for i in range(2, 2 + num_atoms):
            parts = lines[i].strip().split()
            atom = parts[0]
            x, y, z = map(float, parts[1:])
            atoms.append(atom)
            coordinates.append([x, y, z])
        return np.array(atoms), np.array(coordinates)

# Function to compute the center of mass
def compute_center_of_mass(atoms, coordinates):
    masses = {"H": 1.008, "C": 12.01, "O": 16.00, "N": 14.01}  # Extend as needed
    total_mass = sum([masses[atom] for atom in atoms])
    center_of_mass = np.sum([masses[atoms[i]] * coordinates[i] for i in range(len(atoms))], axis=0) / total_mass
    return center_of_mass

# Function to compute the inertia tensor
def compute_inertia_tensor(atoms, coordinates):
    masses = {"H": 1.008, "C": 12.01, "O": 16.00, "N": 14.01}  # Extend as needed
    inertia_tensor = np.zeros((3, 3))
    
    for i in range(len(atoms)):
        mass = masses[atoms[i]]
        r = coordinates[i]
        x, y, z = r[0], r[1], r[2]
        
        inertia_tensor[0, 0] += mass * (y**2 + z**2)
        inertia_tensor[1, 1] += mass * (x**2 + z**2)
        inertia_tensor[2, 2] += mass * (x**2 + y**2)
        inertia_tensor[0, 1] -= mass * x * y
        inertia_tensor[0, 2] -= mass * x * z
        inertia_tensor[1, 2] -= mass * y * z
    
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]
    
    return inertia_tensor

# Function to rotate the coordinates based on the principal moments
def rotate_molecule(atoms, coordinates):
    # Center coordinates by subtracting the center of mass
    center_of_mass = compute_center_of_mass(atoms, coordinates)
    coordinates_centered = coordinates - center_of_mass
    
    inertia_tensor = compute_inertia_tensor(atoms, coordinates_centered)
    
    # Diagonalize the inertia tensor to get eigenvalues (moments) and eigenvectors (axes)
    moments, axes = np.linalg.eigh(inertia_tensor)
    
    # Sort the moments and corresponding axes: largest to smallest
    idx = np.argsort(moments)[::-1]
    principal_moments = moments[idx]
    principal_axes = axes[:, idx]
    
    
    rotation_matrix = np.array([
        principal_axes[:, 2],  
        principal_axes[:, 1],  
        principal_axes[:, 0],  
    ]).T
    
    # Rotate the coordinates
    rotated_coordinates = np.dot(coordinates_centered, rotation_matrix)
    
    # Translate the molecule back to the original center of mass
    return rotated_coordinates + center_of_mass

# Function to save the rotated coordinates to an XYZ file
def save_xyz(file_name, atoms, coordinates):
    with open(file_name, 'w') as file:
        file.write(f"{len(atoms)}\n")
        file.write("Rotated molecule\n")
        for i, atom in enumerate(atoms):
            x, y, z = coordinates[i]
            file.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

# Main function to load, rotate, and save the molecule
def rotate_on_xy(input_file, output_file):
    atoms, coordinates = load_xyz(input_file)
    rotated_coordinates = rotate_molecule(atoms, coordinates)
    save_xyz(output_file, atoms, rotated_coordinates)
    print(f"Rotated molecule saved to {output_file}")

def array_to_string(arr,l):
    stringed = ''
    if len(np.shape(arr)) == 1:
        for i in range(len(arr)):
            stringed += "{:{}f} \n".format(float(arr[i]),l)
    else:
        for i in range(np.shape(arr)[0]):
            for j in range(np.shape(arr)[1]):
                try:
                    stringed += "{:>{}f} \t".format(float(arr[i][j]),l)
                except:
                    stringed += "{:>{}} \t".format(arr[i][j],l)

            stringed += '\n'
    return stringed

parser = argparse.ArgumentParser(description='''
Description: Calculates eigenfunctions and related properties of the Huckel hamiltonian for a given system. 
Inputs:
    - xyz file of molecule (checked for planarity and connectivity)
    - Connectivity matrix (optional - override default)
    - Hückel Hamiltonian (optional - override default)
    - Number of electrons (optional - override default)
Example run: python HuckelSolver.py -xyz molecule.xyz''')

parser.add_argument('-i','--xyz',metavar='',type=str,required=True,help='Structure of molecule as xyz file')
parser.add_argument('-e','--excitations',metavar='',default='',type=str,help='''Text file with list of excitations, 
each line specifies the indexes of the MO involved in the transition. Indexes on the same line must be space separated''')
parser.add_argument('-H','--hamiltonian',type=str,metavar='',default='',help='''Text file with user defined Hückel Hamiltonian, 
the matrix is written in a valid format for the np.loadtxt() function. Use spaces to separate matrix elements''')
parser.add_argument('-d','--cutoff',metavar='',type=float,default=1.6,help='Cutoff distance for identifying neighbouring atoms')
parser.add_argument('-q','--charge',metavar='',type=int,default=0,help='Charge of the system')
parser.add_argument('-M','--mo_size',metavar='',type=int,default=1000,help='Max size of MO lobes (for plotting purposes, default=1000)')
parser.add_argument('-C','--charge_size',metavar='',type=int,default=500,help='Size of Mulliken charges (for plotting purposes, default=500)')
parser.add_argument('-N','--node_size',metavar='',type=int,default=5,help='Font size on atoms (for plotting purposes, default=5)')
parser.add_argument('-E','--edge_size',metavar='',type=int,default=3,help='Font size on bonds (for plotting purposes, default=3)')
parser.add_argument('-R','--reorient',metavar='',type=str,default="True",help='Reorient molecule on xy plane? [True, False] default = True')
parser.add_argument('-T','--text_plot',metavar='',type=str,default="False",help='Write transition properties on plots? [True, False] default = False')
parser.add_argument('-B','--bond_order',metavar='',type=str,default="False",help='Write bond orders on plots? [True, False] default = False')


args = parser.parse_args()

if __name__ == '__main__':

    a = -11.20
    b = -2.62
    q_esu = 4.85 # convert electrostatic unit to Debye

    Elements = ['C','N','O','S','P','F','Cl','Br','I']
    ALPHA = {
        'C':-11.2,
        'N':-12.2
    }
    BETA = {
        'CN':-2.00,
        'CC':-2.62,
        'NN':-2.00,
    }
    filename = args.xyz
    reorient = str2bool(args.reorient)
    if reorient:
        print("Attempting rotaing system coordinate on xy plane")
        rotate_on_xy(filename,filename+'_rotated_xyz')
        filename = filename+'_rotated_xyz'

    else:
        print('Using coordinates from input, no reorientation performed')

    with open(filename, 'r') as f:
        Input = [[num for num in line.strip().split()] for line in f]
        del(Input[0:2])
        Input = np.array(Input)

    w = open(filename+'_output.txt','w')
    w.write('''Hückel Solver
****************
Author: Goran Giudetti
Affiliations: University of Southern California (USC) and University of Groningen (RUG)
****************
    ''')
    mo_size = args.mo_size
    q_size = args.charge_size
    n_font = args.node_size
    e_font = args.edge_size
    text_plot = str2bool(args.text_plot)
    bond_order = str2bool(args.bond_order)

    for i in range(len(Input)-1,-1,-1):
        if Input[i][0] in Elements:
            continue
        else:
            Input = np.delete(Input,i,0)


    Coord = Input[:,1:].astype(dtype=float)
    print(Coord)
    
    Input = np.column_stack((Input[:,0],Coord))

    Labels = Input[:,0].T.tolist()
    nAtoms = len(Coord)
    nElectrons = nAtoms - args.charge

    mu_x = np.zeros((nAtoms,nAtoms))
    mu_y = mu_x.copy()
    mu_z = mu_x.copy()
    for i in range(nAtoms):
        mu_x[i][i] = Input[i][1]
        mu_y[i][i] = Input[i][2]
        mu_z[i][i] = Input[i][3]

    # Occ type objects are density matrices
    Occ = np.zeros((nAtoms,nAtoms))
    if nElectrons % 2 == 0: 
        for i in range(nElectrons//2):
            Occ[i][i] = 2
    else:
        for i in range(nElectrons//2):
            Occ[i][i] = 2
        Occ[i+1][i+1] = 1


    PSI = ConnectMat = np.zeros((nAtoms,nAtoms))
    H = a*np.eye(nAtoms,dtype=float)

    for i in range(nAtoms):
        for j in range(i, nAtoms):
            if i == j:
                H[i][j] = ALPHA[Input[i][0]]
                continue
            dist = abs(np.sqrt((Coord[i][0]-Coord[j][0])**2 + (Coord[i][1]-Coord[j][1])**2  + (Coord[i][2]-Coord[j][2])**2 ))
            if dist <= args.cutoff:
                ConnectMat[i][j] = ConnectMat[j][i] = dist
                try:
                    H[i][j] = H[j][i] = BETA[Input[i][0]+Input[j][0]]
                except:
                    H[i][j] = H[j][i] = BETA[Input[j][0]+Input[i][0]]

    np.savetxt('H_mat.txt',H,fmt='%.2f')
    xyzFile = open(filename+"_effective.xyz", "w")
    xyzFile.write('''{}
                  
{}'''.format(nAtoms,array_to_string(Input,2.5)))
    xyzFile.close()
    #H = test_mat
    if args.hamiltonian != '':
        H = np.loadtxt(args.hamiltonian)
        if (la.issymmetric(H) == False):
            print("ERROR: user-defined Hamiltonian is not symmetric, exiting program")
            quit()
        if len(H) != nAtoms:
            print("ERROR: user-defined Hamiltonian is a {:}x{:} matrix which is inconsistent for a {:} atom system, exiting program".format(len(H),len(H),nAtoms))
            quit()
    Core_pi = np.trace(H)
    #print(H)
    evals,evecs=la.eig(H)
    evals=evals.real
    idx = evals.argsort()#[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx] # Coefficients of atomic orbitals in columns, each column is a molecular orbital
    print("Eval=",evals)
    print("DeltaE=",evals-min(evals))
    E_gs = np.sum((evals-min(evals))@Occ)
    E_gs_tot = np.sum(evals@Occ)
    B_pi = E_gs_tot - Core_pi

    # Creat graph for plotting MOs
    G = nx.Graph()
    posit = {} # the xy coordinates of the atoms are the position of the nodes
    node_labels = {} # Atom element and index in the stripped (without H atoms) molecule are used for labels
    evecs = evecs.T # Transpose coefficients so that each row is an MO
    for i in range(nAtoms):
        G.add_node(i)
        G.nodes[i]['x'] = Coord[i][0]
        G.nodes[i]['y'] = Coord[i][1]
        posit[i] = [G.nodes[i]['x'], G.nodes[i]['y']]
        node_labels[i] = Labels[i]+'-'+str(i+1)
    for i in range(nAtoms-1):
        for j in range(1, nAtoms):
            if ConnectMat[i][j] > 0.0:
                G.add_edge(i, j)

    # Plot MOs

    for i in range(nAtoms):
        colors = []
        sizes = []
        for j in range(len(evals)):
            sizes.append(abs(evecs[i][j])*mo_size)
            if evecs[i][j] > 0.0:
                colors.append("red")
            elif evecs[i][j] < 0.0:
                colors.append("blue")
            else:
                colors.append("grey")

        edges = G.edges()
        nodes = G.nodes()
        f, ax = plt.subplots()
        nx.draw(G, with_labels=True, font_weight='bold', node_size=sizes, font_size=n_font, node_color=colors, 
                pos=posit,labels=node_labels, width=2.0, edgecolors="black")
        limits = plt.axis('off')  # turns off axis
        # Export picture to png
        plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
        plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
        ax.set_aspect('equal', adjustable='box')
        if text_plot:
            plt.text(0.01,0.99, 
'''MO energy = {:.2f} eV
Occupation = {:n}'''.format(evals[i]-min(evals),Occ[i][i]),ha='left', va='top',transform=ax.transAxes
                 )
        plt.savefig(filename+"_N_"+ str(nAtoms)+ "_MO_" + str(i+1)+".png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
    w.write('''
INPUTS
File: {}
N. of atoms: {}
N. of electrons: {}

Removing Hydrogen atoms from molecule, effective coordinates:
{}

Hückel Hamiltonian:
{}

Molecular orbitals eigenenergies:
{}

Atomic orbitals coefficients (row vectors):
{}

    '''.format(filename,nAtoms,nElectrons,array_to_string(Input,2.5),array_to_string(H,5.2),array_to_string(evals,2.3),array_to_string(evecs,2.5)))


    # Computing ground state
    edges_Labels = {}
    mulliken_charges = {}
    mull_charges_array = np.asarray([0.0 for i in range(nAtoms)])
    density_charges = {}
    density_charges_array = np.asarray([0.0 for i in range(nAtoms)])
    Dipole_moment_gs_2 = np.asarray([0.000,0.000,0.000])
    for node in nodes:
        MQ = 0
        for mo in range(nAtoms):
            MQ += Occ[mo][mo]*(evecs[mo][node]**2)
        mulliken_charges[node] = '{:.2f}'.format(1-MQ)
        mull_charges_array[node] = 1-MQ
        density_charges[node] = '{:.2f}'.format(MQ)
        density_charges_array[node] = MQ
    for edge in edges:
        BO = 0
        for mo in range(nAtoms):
            BO += Occ[mo][mo]*evecs[mo][edge[0]]*evecs[mo][edge[1]]
        edges_Labels[edge] = '{:.2f}'.format(BO)
    charges_colors = [float(mulliken_charges[i])for i in G.nodes()]
    charge_labels = {} # Atom element and index in the stripped (without H atoms) plus charge molecule are used for labels
    density_colors = [float(density_charges[i])for i in G.nodes()]
    density_labels = {} # Atom element and index in the stripped (without H atoms) plus density molecule are used for labels
    for i in range(nAtoms):
        charge_labels[i] = node_labels[i]+'\n'+ mulliken_charges[i]
        density_labels[i] = node_labels[i]+'\n'+ density_charges[i]
    charge_sizes = [q_size for i in range(nAtoms)]
    f, ax = plt.subplots()
    nx.draw(G, with_labels=True,node_color=charges_colors , node_size=charge_sizes, font_size=n_font,
             cmap='bwr',vmax=max(mull_charges_array)+0.1,vmin=min(mull_charges_array)-0.1, pos=posit,labels=charge_labels, width=2.0,edgecolors="black")
    if bond_order:
        nx.draw_networkx_edge_labels(G,pos=posit,edge_labels=edges_Labels, font_size=e_font)
    limits = plt.axis('off')  # turns off axis
    # Export picture to png
    plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
    plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
    ax.set_aspect('equal', adjustable='box')
    if text_plot:
        plt.text(0.01,0.99,'''Ground state''',ha='left', va='top',transform=ax.transAxes)
    plt.savefig(filename+"_N_"+ str(nAtoms)+ "_gs.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    #
    # DENSITY PLOT
    #
    f, ax = plt.subplots()
    nx.draw(G, with_labels=True,node_color=density_colors , node_size=charge_sizes, font_size=n_font,
             cmap='bwr',vmax=max(density_charges_array)+0.1,vmin=min(density_charges_array)-0.1, pos=posit,labels=density_labels, width=2.0,edgecolors="black")
    limits = plt.axis('off')  # turns off axis
    # Export picture to png
    plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
    plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
    ax.set_aspect('equal', adjustable='box')
    if text_plot:
        plt.text(0.01,0.99,'''Ground state''',ha='left', va='top',transform=ax.transAxes)
    plt.savefig(filename+"_N_"+ str(nAtoms)+ "_gs_density.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    #
    # DIPOLE MOMENTS
    #
    for i in range(len(mull_charges_array)):
        Dipole_moment_gs_2 += mull_charges_array[i]*Coord[i]
    Dipole_moment_gs_2_tot = np.sqrt(Dipole_moment_gs_2[0]**2+Dipole_moment_gs_2[1]**2+Dipole_moment_gs_2[2]**2)
    w.write('''
Computing ground state properties
Dipole moment = {:.3f} (D) [{:.3f}, {:.3f}, {:.3f}]
Total pi-electron energy = {:.2f} eV
Pi-bonding energy = {:.2f} eV
Mulliken Charges:
{} 

    '''.format(Dipole_moment_gs_2_tot*q_esu,Dipole_moment_gs_2[0]*q_esu,Dipole_moment_gs_2[1]*q_esu,Dipole_moment_gs_2[2]*q_esu,E_gs_tot,B_pi,array_to_string(mull_charges_array,2.3)))
    # Computing excited states
    if args.excitations == '':
        w.close()
        quit()
        
    w.write('''
Computing excited state and transition properties
''')
    mull_charges_array_gs = mull_charges_array.copy()
    density_charges_array_gs = density_charges_array.copy()
    excitations = np.loadtxt(args.excitations,dtype='int')

    if len(np.shape(excitations)) == 1:
        excitations = [excitations]

    count = 0
    for ex in excitations:
        count += 1
        hole = ex[0]-1
        electron = ex[1]-1
        #mu_tr = (abs((np.multiply((1/np.linalg.norm(evecs[:,hole]))*np.outer(evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_x)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_y)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_z)).sum()))
        mu_tr_x = ((np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_x)).sum())
        mu_tr_y = ((np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_y)).sum())
        mu_tr_z = ((np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_z)).sum())
        mu_tr = np.sqrt(mu_tr_x**2 + mu_tr_y**2 + mu_tr_z**2)
        if (ex[0] > nAtoms) or (ex[1] > nAtoms):
            print("Invalid selection, at least 1 index exceeds number of available MOs")
            break
        Occ_ex = Occ.copy()
        if Occ_ex[hole][hole] - 1 < 0:
            print("Invalid selection, cannot excite electron from MO with 0 occupancy")
            break
        if Occ_ex[electron][electron] + 1 > 2:
            print("Invalid selection, cannot promote electron to MO with 2 occupancy")
            break
        Occ_ex[hole][hole] -= 1
        Occ_ex[electron][electron] +=  1
        E_ex = np.sum((evals-min(evals)) @ Occ_ex)
        E_ex_tot = np.sum(evals@Occ_ex)
        B_pi_ex = E_ex_tot - Core_pi


        edges_Labels = {}
        mulliken_charges = {}
        density_charges = {}
        Dipole_moment_ex_2 = np.asarray([0.000,0.000,0.000])
        for node in nodes:
            MQ = 0
            for mo in range(nAtoms):
                MQ += Occ_ex[mo][mo]*(evecs[mo][node]**2)
            mulliken_charges[node] = '{:.2f}'.format(1-MQ-mull_charges_array_gs[node])
            mull_charges_array[node] = 1-MQ-mull_charges_array_gs[node]
            density_charges[node] = '{:.2f}'.format(MQ-density_charges_array_gs[node])
            density_charges_array[node] = MQ-density_charges_array_gs[node]
        for edge in edges:
            BO = 0
            for mo in range(nAtoms):
                BO += Occ_ex[mo][mo]*evecs[mo][edge[0]]*evecs[mo][edge[1]]
            edges_Labels[edge] = '{:.2f}'.format(BO)
        charges_colors = [float(mulliken_charges[i])for i in G.nodes()]
        density_colors = [float(density_charges[i])for i in G.nodes()]

        charge_labels = {} # Atom element and index in the stripped (without H atoms) plus charge molecule are used for labels
        density_labels = {} # Atom element and index in the stripped (without H atoms) plus density molecule are used for labels
        for i in range(nAtoms):
            charge_labels[i] = node_labels[i]+'\n'+ mulliken_charges[i]
            density_labels[i] = node_labels[i]+'\n'+ density_charges[i]
        f, ax = plt.subplots()
        nx.draw(G, with_labels=True,node_color=charges_colors , node_size=charge_sizes, font_size=5,font_weight='bold',
                 cmap='bwr',vmax=max(mull_charges_array)+0.1,vmin=min(mull_charges_array)-0.1, pos=posit,labels=charge_labels, width=2.0,edgecolors="black")
        if bond_order:
            nx.draw_networkx_edge_labels(G,pos=posit,edge_labels=edges_Labels, font_size=3)
        limits = plt.axis('off')  # turns off axis
        # Export picture to png
        plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
        plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
        ax.set_aspect('equal', adjustable='box')
        if text_plot:
            plt.text(0.01,0.99, 
'''Excitation energy = {:.2f} eV
Transition = MO {:n} \u2192 MO {:n}
Tansition dipole m. = {:.3f} (Å)'''.format(E_ex-E_gs,ex[0],ex[1],mu_tr),ha='left', va='top',transform=ax.transAxes
                 )
        plt.savefig(filename+"_N_"+ str(nAtoms)+ "_transition_" + str(ex[0])+"_"+str(ex[1])+".png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        #
        # DENSITY PLOT
        #
        f, ax = plt.subplots()
        nx.draw(G, with_labels=True,node_color=density_colors , node_size=charge_sizes, font_size=5,font_weight='bold',
                 cmap='bwr',vmax=max(density_charges_array)+0.1,vmin=min(density_charges_array)-0.1, pos=posit,labels=density_labels, width=2.0,edgecolors="black")
        limits = plt.axis('off')  # turns off axis
        # Export picture to png
        plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
        plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
        ax.set_aspect('equal', adjustable='box')
        if text_plot:
            plt.text(0.01,0.99, 
'''Excitation energy = {:.2f} eV
Transition = MO {:n} \u2192 MO {:n}
Tansition dipole m. = {:.3f} (Å)'''.format(E_ex-E_gs,ex[0],ex[1],mu_tr),ha='left', va='top',transform=ax.transAxes
                 )
        plt.savefig(filename+"_N_"+ str(nAtoms)+ "_transition_density_" + str(ex[0])+"_"+str(ex[1])+".png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        #
        # DIPOLE 2
        #
        for i in range(len(mull_charges_array)):
            Dipole_moment_ex_2 += mull_charges_array[i]*Coord[i]
        Dipole_moment_ex_2_tot = np.sqrt(Dipole_moment_ex_2[0]**2+Dipole_moment_ex_2[1]**2+Dipole_moment_ex_2[2]**2)
        w.write('''********
Excited state {}
Excitation energy = {:.2f} eV
Total pi-electron energy = {:.2f} eV
Pi-bonding energy = {:.2f} eV   
Transition = MO {:n} \u2192 MO {:n}
Dipole moment = {:.3f} (D) [{:.3f}, {:.3f}, {:.3f}]
Diff. dipole m. = {:.3f} (D)
Transition dipole m. = {:.3f} (Å)
Mulliken Charges:
{}
'''.format(count,E_ex-E_gs,E_ex_tot,B_pi_ex,ex[0],ex[1],Dipole_moment_ex_2_tot*q_esu,Dipole_moment_ex_2[0]*q_esu,Dipole_moment_ex_2[1]*q_esu,Dipole_moment_ex_2[2]*q_esu,(Dipole_moment_ex_2_tot-Dipole_moment_gs_2_tot)*q_esu,mu_tr,array_to_string(mull_charges_array,2.3)))
    w.write('''********''')
    w.close()
    unique, counts = np.unique(H, return_counts=True)
    print("Counting parameters in hamiltonian: {parameter: count}\n",dict(zip(unique, counts)))
