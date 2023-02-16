"""
Created on Thu Feb 2 15:00:00 2023

@author: Goran Giudetti

Description:
This script uses excitation energies and oscillator strengths from a Q-Chem TD-DFT/CIS output to generate an absorption/emission spectrum by convolution of Gaussian functions. If multiple excitated state calculations are provided (as in excited state AIMD), the spectra is averaged through each job.
Simply provide the path to your Q-Chem job in the 'file' (string) variable; one has to adjust the values of 'start', 'finish', 'points', 'width' as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *

#Plot grid in eV
start = 2
finish = 6
points = 10000

# Define the FWHM (width) in eV
width = 0.1
sigma = width/(2*np.sqrt(log(2)))

# Excitation energies in eV and oscillator strengths
ecc = []
osc = []

# Read Q-Chem output and store excitations with stregths
o = open("Q-Chem_output.out", "r")
rl = o.readlines()
o.close()


for i in range(len(rl)):
    if "cis_n_roots" in rl[i].lower():
        div = float(rl[i].strip().split()[-1])
    if "excitation energy (eV)" in rl[i]:
        ecc.append(float(rl[i].strip().split()[-1]))
    if "Strength   : " in rl[i]:
        osc.append(float(rl[i].strip().split()[-1]))


print("Starting plotting")


def GaussSpectrum(x, band, strength, sigma):
    "Return a normalized Gaussian, input of energy is eV"
    GaussBand = strength * ( 1/(sigma*np.sqrt(2*np.pi))) * np.exp(- ((x-band)** 2)/(2*sigma**2))
    return GaussBand

x = np.linspace(start, finish, points)


spectrum = 0


print(len(ecc)/div)

for count, peak in enumerate(ecc):
    GaussPeak = GaussSpectrum(x, peak, osc[count], sigma)/len(ecc)/div
    spectrum += GaussPeak


#Export data in csv
w = open("Bands.csv", "w")
w.write("BAND(eV),BAND(nm),OSC.\n")
for i in range(len(ecc)):
    w.write("{:.4f},{:.4f},{:.4f}\n".format(ecc[i],1240/ecc[i],osc[i]))
w.close()

w = open("Spectra.csv", "w")
w.write("BAND(eV),BAND(nm),OSC.\n")
for i in range(len(x)):
    w.write("{:.4f},{:.4f},{:.4f}\n".format(x[i],1240/x[i],spectrum[i]))
w.close()

#Export spectrum in eV
fig, ax = plt.subplots()
plt.xlabel('eV')
plt.ylabel('Intensity')
ax.plot(x, spectrum, label='Your spectrum')
plt.legend()
plt.title('Your title')
plt.savefig('Your_image_eV.png', format='png', dpi=300, bbox_inches='tight')

#Export spectrum in nm
x1 = [1240/z for z in x] # Convert eV to nm
fig, ax = plt.subplots()
plt.xlabel('$\lambda$ / nm')
plt.ylabel('Intensity')
ax.plot(x1, spectrum, label='Your spectrum')
plt.legend()
plt.title('Your title')
plt.savefig('Your_image_nm.png', format='png', dpi=300, bbox_inches='tight')

plt.show()
