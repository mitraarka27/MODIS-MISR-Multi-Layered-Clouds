#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:15:40 2020

@author: arka
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib

global h

lookup = np.loadtxt('MODIS_StdUSAtm_props.txt')
Pres=lookup[:,0]
Temp=lookup[:,1]
z=lookup[:,2]
transmittance=np.zeros((len(Pres),4))
wt_fn=np.zeros((len(Pres),4))
planck=np.zeros((len(Pres),4))
transmittance[:,0]=lookup[:,3];transmittance[:,1]=lookup[:,6]
transmittance[:,2]=lookup[:,9];transmittance[:,3]=lookup[:,12]
wt_fn[:,0]=lookup[:,4];wt_fn[:,1]=lookup[:,7]
wt_fn[:,2]=lookup[:,10];wt_fn[:,3]=lookup[:,13]
planck[:,0]=lookup[:,5];planck[:,1]=lookup[:,8]
planck[:,2]=lookup[:,11];planck[:,3]=lookup[:,14]
h = float(np.log(Pres[len(Pres)-1]) - np.log(Pres[0])) / len(Pres)
Ics = np.zeros(4)
intensity_at_level=np.zeros((len(Pres)-40,5))
ratio_at_level=np.zeros((len(Pres)-40,4))
intensity_at_level[:,4]=[Pres[j] for j in range(20,len(Pres)-20)]
ratio_at_level[:,3]=[Pres[j] for j in range(20,len(Pres)-20)]
trial_pressure=np.zeros((1000,8))
trial_ratio=np.zeros((1000,4))

def trapezoidal(f, n):
    s = 0.0
    s += f[0]/2.0
    for i in range(1, n-1):
        s += f[i]
    s += f[n-1]/2.0
    return s * h

def I_integral(B,wt_fn,wv_n):
    return(B[:,wv_n]*wt_fn[:,wv_n])
    
for wv in range(0,4):
    f = I_integral(planck,wt_fn,wv)
    integral = trapezoidal(f, n=len(Pres))
    Ics[wv] = (planck[len(Pres)-1,wv]*transmittance[len(Pres)-1,wv] + 
       integral) * 1e-6
#    print(Ics[wv])
    for j in range (20,len(Pres)-20):
        f = I_integral(planck,wt_fn,wv)
        integral = trapezoidal(f, n=j)
        intensity_at_level[j-20,wv] = (planck[len(Pres)-1,wv]*
                          transmittance[len(Pres)-1,wv] + integral)*1e-6


for wv in range(0,3):
    for j in range (20,len(Pres)-20):
        ratio_at_level[j-20,wv] = (intensity_at_level[j-
                      20,wv+1]-Ics[wv+1])/(intensity_at_level[j-20,wv]-Ics[wv])
    
plt.figure(figsize=(7,7))
ax = plt.gca()
plt.plot(ratio_at_level[:,0],ratio_at_level[:,3], label='$R_{2/1}$')
plt.plot(ratio_at_level[:,1],ratio_at_level[:,3], label='$R_{3/2}$')
plt.plot(ratio_at_level[:,2],ratio_at_level[:,3], label='$R_{4/3}$')
plt.text(0.8,900,' 1 - 13.3 $\mu m$ \n \
2 - 13.6 $\mu m$ \n \
3 - 13.9 $\mu m$ \n \
4 - 14.2 $\mu m$ \n ', fontsize=13)
ax.invert_yaxis()
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel(r'Radiance Ratio , $R_{j/i}$ ($P_{c}$)', fontsize=14)
plt.ylabel('Cloud Top Pressure, $P_{c}$ (hPa)',fontsize=14)
plt.legend(fontsize=13, loc='best')
plt.savefig('./plots/MODIS_CO2bands_RadRatios_StdUSAtm.png', dpi=400)
plt.show()   


trial_pressure[:,4] = np.min(intensity_at_level[:,4])+ \
           np.random.rand(1000)*(np.max(intensity_at_level[:,4])-np.min(intensity_at_level[:,4]))

f1 = interp1d(intensity_at_level[:,4],intensity_at_level[:,0])

f2 = interp1d(intensity_at_level[:,4],intensity_at_level[:,1])

f3 = interp1d(intensity_at_level[:,4],intensity_at_level[:,2])

f4 = interp1d(intensity_at_level[:,4],intensity_at_level[:,3])

trial_pressure[:,0]=f1(trial_pressure[:,4])
trial_pressure[:,1]=f2(trial_pressure[:,4])
trial_pressure[:,2]=f3(trial_pressure[:,4])
trial_pressure[:,3]=f4(trial_pressure[:,4])

for ntrial in range(0,1000):
    for wv in range(0,3):
        trial_ratio[ntrial,wv]=(trial_pressure[ntrial,wv+1]-Ics[wv+1])/(trial_pressure[ntrial,wv]-Ics[wv])

for ntrial in range(0,1000):
    for wv in range(0,3):
        z = trial_ratio[ntrial,wv]-ratio_at_level[:,wv]
        asign = np.sign(z)
        signchange = (np.abs((np.roll(asign, 1) - asign)) != 0).astype(int)
        signchange[0]=0
        obs_pres=intensity_at_level[np.min(np.where(signchange==1)),4]
        trial_pressure[ntrial,wv+5]=obs_pres

