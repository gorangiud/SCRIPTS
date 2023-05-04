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
import matplotlib.pyplot as plt
import networkx as nx
import argparse

def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
     
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    #print(f"equation of plane is {a}x + {b}y + {c}z + {d} = 0.")
    return(a,b,c,d)

def rotate_on_xy(points):
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        print("Points selected are collinear, rotation algorithm won't work. No rotation applied.")
        return points
    a,b,c,d = equation_plane(points[0][0], points[0][1],points[0][2],points[1][0],points[1][1],points[1][2],points[2][0],points[2][1],points[2][2])

    tras = -d/c
    cos_th = c/np.sqrt(a**2 + b**2 + c**2)
    sin_th = np.sqrt((a**2 + b**2)/(a**2 + b**2 + c**2))
    u_1 = b/np.sqrt(a**2 + b**2 )
    u_2 = -a/np.sqrt(a**2 + b**2 )

    rot_mat = [
        [cos_th + (1-cos_th)*u_1**2, u_1*u_2*(1-cos_th), u_2*sin_th],
        [u_1*u_2*(1-cos_th), cos_th + (1-cos_th)*u_2**2, -u_1*sin_th],
        [-u_2*sin_th, u_1*sin_th, cos_th]
    ]


    rotated_points = (rot_mat@points.T).T

    return(rotated_points)

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
parser.add_argument('-q','--charge',metavar='',type=int,default=0,help='Charge of the system')

args = parser.parse_args()

if __name__ == '__main__':

    a = -11.2
    b = -2.62


    Elements = ['C','N','O','S','P','F','Cl','Br','I']
    ALPHA = {
        'C':-11.2,
        'N':-12.2
    }
    BETA = {
        'CN':-2.00,
        'CC':-2.62
    }
    filename = args.xyz

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


    for i in range(len(Input)-1,-1,-1):
        if Input[i][0] in Elements:
            continue
        else:
            Input = np.delete(Input,i,0)


    Coord = Input[:,1:].astype(dtype=float)
    print(Coord)
    print("Attempting rotaing system coordinate on xy plane")
    Coord = rotate_on_xy(Coord).astype(dtype=float)
 
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
    if nElectrons % 2 == 0:
        Occ = np.zeros((nAtoms,nAtoms))
        for i in range(nElectrons//2):
            Occ[i][i] = 2
    else:
        Occ = np.zeros((nAtoms,nAtoms))
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
            if dist <= 1.6:
                ConnectMat[i][j] = ConnectMat[j][i] = dist
                try:
                    H[i][j] = H[j][i] = BETA[Input[i][0]+Input[j][0]]
                except:
                    H[i][j] = H[j][i] = BETA[Input[j][0]+Input[i][0]]

    np.savetxt('H_mat.txt',H,fmt='%.2f')
    
    #H = test_mat
    if args.hamiltonian != '':
        H = np.loadtxt(args.hamiltonian)
    print(H)
    evals,evecs=la.eig(H)
    evals=evals.real
    idx = evals.argsort()#[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx] # Coefficients of atomic orbitals in columns, each column is a molecular orbital
    print("Eval=",evals)
    print("DeltaE=",evals-min(evals))
    E_gs = np.sum((evals-min(evals))@Occ)

    Dipole_MO = np.array([0.0 for i in range(nAtoms)])

    for i in range(nAtoms):
        Dipole_MO[i] = (np.multiply(np.outer((1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T,(1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T), mu_x)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T,(1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T), mu_y)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T,(1/np.linalg.norm(evecs[:,i]))*evecs[:,i].T), mu_z)).sum()
    print(Dipole_MO)


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
            sizes.append(abs(evecs[i][j])*1000)
            if evecs[i][j] > 0.0:
                colors.append("red")
            elif evecs[i][j] < 0.0:
                colors.append("blue")
            else:
                colors.append("grey")

        edges = G.edges()
        nodes = G.nodes()
        f, ax = plt.subplots()
        nx.draw(G, with_labels=True, font_weight='bold', node_size=sizes, font_size=5, node_color=colors, 
                pos=posit,labels=node_labels, width=2.0)
        limits = plt.axis('off')  # turns off axis
        # Export picture to png
        plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
        plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
        ax.set_aspect('equal', adjustable='box')
        plt.text(0.01,0.99, 
'''MO energy = {:.2f} eV
Occupation = {:n}'''.format(evals[i]-min(evals),Occ[i][i]),ha='left', va='top',transform=ax.transAxes
                 )
        plt.savefig(filename+"_N_"+ str(nAtoms)+ "_MO_" + str(i+1)+".png", format='png', dpi=300, bbox_inches='tight')
        plt.close
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

Molecular orbitals relative eigenenergies:
{}

Atomic orbitals coefficients (row vectors):
{}

    '''.format(filename,nAtoms,nElectrons,array_to_string(Input,2.5),array_to_string(H,5.2),array_to_string(evals-min(evals),2.3),array_to_string(evecs,2.5)))


    # Computing ground state
    edges_Labels = {}
    mulliken_charges = {}
    for node in nodes:
        MQ = 0
        for mo in range(nAtoms):
            MQ += Occ[mo][mo]*evecs[mo][node]**2
        mulliken_charges[node] = '{:.2f}'.format(1-MQ)
    for edge in edges:
        BO = 0
        for mo in range(nAtoms):
            BO += Occ[mo][mo]*evecs[mo][edge[0]]*evecs[mo][edge[1]]
        edges_Labels[edge] = '{:.2f}'.format(BO)
    charges_colors = [float(mulliken_charges[i])for i in G.nodes()]
    charge_labels = {} # Atom element and index in the stripped (without H atoms) plus charge molecule are used for labels
    for i in range(nAtoms):
        charge_labels[i] = node_labels[i]+'\n'+ mulliken_charges[i]
    charge_sizes = [500 for i in range(nAtoms)]
    f, ax = plt.subplots()
    nx.draw(G, with_labels=True,node_color=charges_colors , node_size=charge_sizes, font_size=5,
             cmap='coolwarm',vmax=1,vmin=-1, pos=posit,labels=charge_labels, width=2.0)
    nx.draw_networkx_edge_labels(G,pos=posit,edge_labels=edges_Labels, font_size=3)
    limits = plt.axis('off')  # turns off axis
    # Export picture to png
    plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
    plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
    ax.set_aspect('equal', adjustable='box')
    plt.text(0.01,0.99,'''Ground state''',ha='left', va='top',transform=ax.transAxes)
    plt.savefig(filename+"_N_"+ str(nAtoms)+ "_gs.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    Dipole_gs = np.sum(Dipole_MO @ Occ)
    w.write('''
Computing ground state properties
Dipole moment = {:.3f} (Å)

    '''.format(Dipole_gs))
    # Computing excited states
    if args.excitations == '':
        w.close()
        quit()
        
    w.write('''
Computing excited state and transition properties
''')
    excitations = np.loadtxt(args.excitations,dtype='int')

    if len(np.shape(excitations)) == 1:
        excitations = [excitations]

    count = 0
    for ex in excitations:
        count += 1
        hole = ex[0]-1
        electron = ex[1]-1
        mu_tr = (abs((np.multiply((1/np.linalg.norm(evecs[:,hole]))*np.outer(evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_x)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_y)).sum() + (np.multiply(np.outer((1/np.linalg.norm(evecs[:,hole]))*evecs[:,hole].T,(1/np.linalg.norm(evecs[:,electron]))*evecs[:,electron].T), mu_z)).sum()))
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


        edges_Labels = {}
        mulliken_charges = {}
        for node in nodes:
            MQ = 0
            for mo in range(nAtoms):
                MQ += Occ_ex[mo][mo]*evecs[mo][node]**2
            mulliken_charges[node] = '{:.2f}'.format(1-MQ)
        for edge in edges:
            BO = 0
            for mo in range(nAtoms):
                BO += Occ_ex[mo][mo]*evecs[mo][edge[0]]*evecs[mo][edge[1]]
            edges_Labels[edge] = '{:.2f}'.format(BO)
        charges_colors = [float(mulliken_charges[i])for i in G.nodes()]

        charge_labels = {} # Atom element and index in the stripped (without H atoms) plus charge molecule are used for labels
        for i in range(nAtoms):
            charge_labels[i] = node_labels[i]+'\n'+ mulliken_charges[i]
        charge_sizes = [500 for i in range(nAtoms)]
        f, ax = plt.subplots()
        nx.draw(G, with_labels=True,node_color=charges_colors , node_size=charge_sizes, font_size=5,
                 cmap='coolwarm',vmax=1,vmin=-1, pos=posit,labels=charge_labels, width=2.0)
        nx.draw_networkx_edge_labels(G,pos=posit,edge_labels=edges_Labels, font_size=3)
        limits = plt.axis('off')  # turns off axis
        # Export picture to png
        plt.xlim((min(Coord[:,0])-1.397, max(Coord[:,0])+1.397))
        plt.ylim((min(Coord[:,1])-1.397, max(Coord[:,1])+1.397))
        ax.set_aspect('equal', adjustable='box')
        plt.text(0.01,0.99, 
'''Excitation energy = {:.2f} eV
Transition = MO {:n} \u2192 MO {:n}
Tansition dipole m. = {:.3f} (Å)'''.format(E_ex-E_gs,ex[0],ex[1],mu_tr),ha='left', va='top',transform=ax.transAxes
                 )
        plt.savefig(filename+"_N_"+ str(nAtoms)+ "_transition_" + str(ex[0])+"_"+str(ex[1])+".png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        Dipole_ex = np.sum(Dipole_MO @ Occ_ex)
        w.write('''********
Excited state {}
Excitation energy = {:.2f} eV   
Transition = MO {:n} \u2192 MO {:n}
Dipole moment = {:.3f} (Å)
Diff. dipole m. = {:.3f} (Å)
Tansition dipole m. = {:.3f} (Å)
'''.format(count,E_ex-E_gs,ex[0],ex[1],Dipole_ex,Dipole_ex-Dipole_gs,mu_tr))
    w.write('''********''')
    w.close()
