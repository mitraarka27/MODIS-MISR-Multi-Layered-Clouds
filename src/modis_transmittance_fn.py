#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:20:21 2020

This program calculates the transmittances and subsequently the weighting functions of 
the 4 MODIS CO2 absorption bands (13.3, 13.6, 13.9 and 14.2 um) and 11um and 12um. The Hitran 
Python Interface (hapi) is used to calculate the coefficients of absorption over the bands and 
then the transmittance function is is calculated from those coefficients, for each
pressure/temperature level and gas concentration as provided in the Standard US Atmosphere. 
The releveant lookup tables for MODIS CTP estimation are output to files and the weighting 
functions are plotted. Gaseous absorption comes from CO2, H20, O2 and O3. Transmission function 
also takes into account the effect of varying the atmospheric temperature. To interpolate to a 
higher resolution in Pressure, Temperature and Heights are interpolated from the 40 given pressure 
levels to 200 equidistant (in P coordinates) levels, assuming that T is a linear function of
log(Pressure).

@author: Arka Mitra
"""

import hapi
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib

h = 6.626e-34
c = 299792458
kB = 1.38e-23

atm_file = np.loadtxt('AFGLUS.70KM',skiprows=2) #US Standard Atmosphere, 1976. ( AFGL-TR-86-0110)  
out = open('MODIS_StdUSAtm_props.txt', 'w')

def get_ext(p,T,z,gas): 
    #Function to calculate transmittance from gaseous absorption over a MODIS band
    #for a given P and T level
    nu = hapi.getColumn(gas,'nu')
    nu, coeff_abs = hapi.absorptionCoefficient_Lorentz(SourceTables=gas,
                                                 Environment={'p': p/1013.,'T': T},
                                                 OmegaGrid=nu, HITRAN_units=False,
                                                 GammaL='gamma_self')
    nu,transm = hapi.transmittanceSpectrum(nu,coeff_abs,
                                     Environment={'l':20.})
    return np.mean(transm)

def planck(wav, T):
    #Function to calculate Planck radiation for emissivity=1, given wavelength and Temp
    a = 2.0*h*c**2
    b = h*c/(wav*kB*T)
    intensity = a/((wav**5)*(np.exp(b)-1.0))
    return intensity

def dt_dlnP(P,t):
    #Function to calculate normalized wighting functions for a MODIS band, given P and 
    #transmittance profiles
    dtdlnP = np.zeros(len(P))
    for i in range(len(P)-1):
        dtdlnP[i] = -(t[i]-t[i+1])/(np.log(P[i])-np.log(P[i+1]))
    dtdlnP[len(P)-1] = dtdlnP[i] 
    wt_fn = [(dtdlnP[i]-np.min(dtdlnP))/(np.max(dtdlnP)-np.min(dtdlnP))
             for i in range(len(dtdlnP))]
    return wt_fn

def T_fn(Pres,Temp,z):
    #Function that calculates transmittances for all Pressure levels
    transmittance = np.zeros(len(Pres))
    transmittance[len(Pres)-1] = 0.
    for i in range(0,len(Pres)):
        mol_ext =  np.mean(get_ext(Pres[i],Temp[i],z,'CO2')) * \
                   np.mean(get_ext(Pres[i],Temp[i],z,'O2')) * \
                   np.mean(get_ext(Pres[i],Temp[i],z,'O3')) * \
                   np.mean(get_ext(Pres[i],Temp[i],z,'H2O')) 
        transmittance[i] = mol_ext
    return transmittance
    

Pres = np.linspace(0.2,1000.,500)
fT = interp1d(np.log(atm_file[:,1]),atm_file[:,2])
Temp = fT(np.log(Pres))
fT = interp1d(np.log(atm_file[:,1]),atm_file[:,0])
z = fT(np.log(Pres))
transmittance=np.zeros((len(Pres),6))
wt_fn=np.zeros((len(Pres),6))

for i in range(0,5):
    if (i==0):
        u=741.56
        l=758.44
    elif (i==1):
        u=725.43
        l=741.56
    elif (i==2):
        u=709.98
        l=725.38
    elif (i==3):
        u=695.17
        l=709.98
    elif (i==4):
        u=886.52
        l=927.64
    else:
        u=815.00
        l=849.62
    hapi.fetch('CO2',2,1,u,l) # 13.3-751.87,751.89, 13.6-735.26,735.30, 13.9-719.2,719.44 14.4-704.22,704.24
    hapi.fetch('O3',2,1,u,l)
    hapi.fetch('O2',2,1,u,l)
    hapi.fetch('H2O',2,1,u,l)
    hapi.fetch('NO2',2,1,u,l)
    transmittance[:,i] = T_fn(Pres, Temp, z)
    wt_fn[:,i] = dt_dlnP(Pres,transmittance[:,i])
    
for i in range(0,len(Pres)):
    out.write(str(Pres[i])+'  '+str(Temp[i])+'  '+str(z[i])+'  '+
              str(transmittance[i,0])+'  '+str(wt_fn[i,0])+'  '+str(planck(13.335*1e-6,Temp[i]))+'  '+
              str(transmittance[i,1])+'  '+str(wt_fn[i,1])+'  '+str(planck(13.635*1e-6,Temp[i]))+'  '+
              str(transmittance[i,2])+'  '+str(wt_fn[i,2])+'  '+str(planck(13.935*1e-6,Temp[i]))+'  '+
              str(transmittance[i,3])+'  '+str(wt_fn[i,3])+'  '+str(planck(14.235*1e-6,Temp[i]))+'  '+
              str(transmittance[i,4])+'  '+str(wt_fn[i,4])+'  '+str(planck(11.200*1e-6,Temp[i]))+'  '+
              str(transmittance[i,5])+'  '+str(wt_fn[i,5])+'  '+str(planck(12.000*1e-6,Temp[i]))+'  '+'\n')
out.close()

#Plot out the weighting functions wrt pressure
plt.figure(figsize=(7,7))
ax = plt.gca()
plt.plot(wt_fn[:,0],Pres, label='$\lambda$ = 13.3 $\mu m$')
plt.plot(wt_fn[:,1],Pres, label='$\lambda$ = 13.6 $\mu m$')
plt.plot(wt_fn[:,2],Pres, label='$\lambda$ = 13.9 $\mu m$')
plt.plot(wt_fn[:,3],Pres, label='$\lambda$ = 14.2 $\mu m$')
plt.plot(wt_fn[:,4],Pres, label='$\lambda$ = 11.2 $\mu m$')
# plt.plot(wt_fn[:,5],Pres, label='$\lambda$ = 12.0 $\mu m$')
ax.invert_yaxis()
ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel(r'Normalized weighting functions, $\frac{d\tau}{dln P}$', fontsize=14)
plt.ylabel('Pressure (hPa)',fontsize=14)
plt.legend(fontsize=13, loc='best')
plt.savefig('./plots/MODIS_CO2bands_WeightingFns_StdUSAtm.png', dpi=400)
plt.show()   
