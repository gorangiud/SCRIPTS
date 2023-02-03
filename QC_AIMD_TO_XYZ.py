"""
Created on Thu Jan 26 16:41:00 2023

@author: Goran Giudetti

Description:
This script creates a xyz trajectory from a Q-Chem aimd output file.
Simply provide the path to your Q-Chem job in the 'file' (string) variable; the trajectory will be in the same path and with the same name of the Q-Chem job, but the .xyz extension will be added.
"""

file = "Path to Q-Chem AIMD job"

o = open(file,"r")
w = open(file+".xyz","w")
rl = o.readlines()
o.close()
i1 = rl.index("$molecule\n")
i2 = rl.index("$end\n")
t_au = 0.0242 # femtoseconds
mol = i2 - i1 - 2
t, z = 0, 0

for line in rl:
    z += 1
    if "time_step" in line.lower():
        ts = float(line.strip().split()[-1])
    if "Standard Nuclear Orientation" in line:
        w.write(str(mol)+ "\n")
        w.write("Time step: {} ({} fs)\n".format(ts*t,ts*t*t_au))
        for i in range(z+2,z+2+mol):
            w.write(rl[i][10:])
        t += 1
w.close()
