# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:16:29 2020

@author: Arka Mitra
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from scipy.stats import norm

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
ratio_at_level=np.zeros((len(Pres)-40,4,20))
intensity_at_level[:,4]=[Pres[j] for j in range(20,len(Pres)-20)]
trial_pressure=np.zeros((5000,8))
trial_ratio=np.zeros((5000,4))
band_snr=[0.75,1.0,1.0,1.25]

#Now let's simulate the satellite-observed TOA radiances
sat_radiances=np.zeros((len(Pres)-40,6,20))
sat_ratio=np.zeros((len(Pres)-40,5,20))
trial_sat_pressure=np.zeros((5000,18,20))
trial_sat_ratio=np.zeros((5000,3,20))

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
    for j in range (20,len(Pres)-20):
        f = I_integral(planck,wt_fn,wv)
        integral = trapezoidal(f, n=j)
        intensity_at_level[j-20,wv] = (planck[len(Pres)-1,wv]*
                          transmittance[len(Pres)-1,wv] + integral)*1e-6
        sat_radiances[j-20,4,:] = Pres[j]
        eff_amount=0.
        for nAmount in range(0,20):
            eff_amount += 0.05
            sat_radiances[j-20,wv,nAmount] = Ics[wv]+eff_amount* \
                (intensity_at_level[j-20,wv]-Ics[wv])
            sat_radiances[j-20,5,nAmount] = eff_amount
            

#For black clouds (nAmount=20, e=1) and non-black clouds 
#Calculated radiance ratios at known pressure levels
for wv in range(0,3):
    for j in range (20,len(Pres)-20):
        for nAmount in range(0,20):
            ratio_at_level[j-20,3,nAmount] = Pres[j]
            ratio_at_level[j-20,wv,nAmount] = (sat_radiances[j-\
                          20,wv+1,nAmount]-Ics[wv+1])/(sat_radiances[j-20,wv,nAmount]-Ics[wv]) 
    
#Ratios reamin the same irrest+pective of cloud amounts -
#The true advantage of this method!
    
#plt.figure(figsize=(7,7))
#ax = plt.gca()
#plt.plot(ratio_at_level[:,0,20],ratio_at_level[:,3,20], label='$R_{2/1}$')
#plt.plot(ratio_at_level[:,1,20],ratio_at_level[:,3,20], label='$R_{3/2}$')
#plt.plot(ratio_at_level[:,2,20],ratio_at_level[:,3,20], label='$R_{4/3}$')
#plt.text(0.8,900,' 1 - 13.3 $\mu m$ \n \
#2 - 13.6 $\mu m$ \n \
#3 - 13.9 $\mu m$ \n \
#4 - 14.2 $\mu m$ \n ', fontsize=13)
#ax.invert_yaxis()
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.xlabel(r'Radiance Ratio , $R_{j/i}$ ($P_{c}$)', fontsize=14)
#plt.ylabel('Cloud Top Pressure, $P_{c}$ (hPa)',fontsize=14)
#plt.legend(fontsize=13, loc='best')
##plt.savefig('./plots/MODIS_CO2bands_RadRatios_StdUSAtm.png', dpi=400)
#plt.show()   


trial_sat_pressure[:,4,19] = np.min(sat_radiances[:,4,19])+ \
           np.random.rand(5000)*(np.max(sat_radiances[:,4,19])-\
                          np.min(sat_radiances[:,4,19]))

f1 = interp1d(sat_radiances[:,4,19],sat_radiances[:,0,19])

f2 = interp1d(sat_radiances[:,4,19],sat_radiances[:,1,19])

f3 = interp1d(sat_radiances[:,4,19],sat_radiances[:,2,19])

f4 = interp1d(sat_radiances[:,4,19],sat_radiances[:,3,19])

#Approximate cloudy radiance at trial pressure levels
trial_sat_pressure[:,0,19]=f1(trial_sat_pressure[:,4,19])
trial_sat_pressure[:,1,19]=f2(trial_sat_pressure[:,4,19])
trial_sat_pressure[:,2,19]=f3(trial_sat_pressure[:,4,19])
trial_sat_pressure[:,3,19]=f4(trial_sat_pressure[:,4,19])

eff_amount=0.
for nAmount in range(0,20):
    eff_amount += 0.05
    trial_sat_pressure[:,4,nAmount]=trial_sat_pressure[:,4,19]
    trial_sat_pressure[:,5,nAmount]=eff_amount
    f1 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,0,nAmount])
    
    f2 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,1,nAmount])
    
    f3 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,2,nAmount])
    
    f4 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,3,nAmount])

#Simulated satellite observed radiances at each trial pressure level for
#given values of cloud emissivities, in non-black cases.
    trial_sat_pressure[:,0,nAmount]=f1(trial_sat_pressure[:,4,nAmount])
    trial_sat_pressure[:,1,nAmount]=f2(trial_sat_pressure[:,4,nAmount])
    trial_sat_pressure[:,2,nAmount]=f3(trial_sat_pressure[:,4,nAmount])
    trial_sat_pressure[:,3,nAmount]=f4(trial_sat_pressure[:,4,nAmount])

#Let's add Gaussian noise to satellite signals
for nAmount in range(0,20):
            noise = np.random.normal(0, 0.25, 5000)
            trial_sat_pressure[:,0,nAmount]=trial_sat_pressure[:,0,nAmount]+noise
            noise = np.random.normal(0, 0.25, 5000)
            trial_sat_pressure[:,1,nAmount]=trial_sat_pressure[:,1,nAmount]+noise
            noise = np.random.normal(0, 0.25, 5000)
            trial_sat_pressure[:,2,nAmount]=trial_sat_pressure[:,2,nAmount]+noise
            noise = np.random.normal(0, 0.25, 5000)
            trial_sat_pressure[:,3,nAmount]=trial_sat_pressure[:,3,nAmount]+noise
            

for ntrial in range(0,5000):
    for wv in range(0,3):
        for nAmount in range(0,20):
#Observed radiance ratio
            if (trial_sat_pressure[ntrial,wv+1,nAmount]- Ics[wv+1]) < 2*band_snr[wv+1] and \
                (trial_sat_pressure[ntrial,wv,nAmount]-Ics[wv]) < 2*band_snr[wv]:
                    trial_sat_ratio[ntrial,wv,nAmount]=(trial_sat_pressure[ntrial,wv+1,nAmount]-\
                               Ics[wv+1])/(trial_sat_pressure[ntrial,wv,nAmount]-Ics[wv]) 

for ntrial in range(0,5000):
    for wv in range(0,3):
        for nAmount in range(0,20):
            z = trial_sat_ratio[ntrial,wv,nAmount]-ratio_at_level[:,wv,nAmount]
            try:
                asign = np.sign(z)
                signchange = (np.abs((np.roll(asign, 1) - asign)) != 0).astype(int)
                signchange[0]=0
                obs_pres=sat_radiances[np.min(np.where(signchange==1)),4,nAmount]
                if obs_pres>900. or obs_pres<100.:
                    obs_pres=np.nan
                trial_sat_pressure[ntrial,wv+6,nAmount]=obs_pres
                trial_sat_pressure[ntrial,wv+9,nAmount]= \
                    obs_pres-trial_sat_pressure[ntrial,4,nAmount]
                trial_sat_pressure[ntrial,wv+12,nAmount]= \
                    (trial_sat_pressure[ntrial,wv,nAmount]-Ics[wv])/ \
                    (trial_sat_pressure[ntrial,wv,19]-Ics[wv])
                trial_sat_pressure[ntrial,wv+15,nAmount]= \
                    (trial_sat_pressure[ntrial,wv+11,nAmount]- \
                     trial_sat_pressure[ntrial,5,nAmount])
            except:
                trial_sat_pressure[ntrial,wv+6,nAmount]=np.nan
                trial_sat_pressure[ntrial,wv+9,nAmount]=np.nan
                trial_sat_pressure[ntrial,wv+12,nAmount]=np.nan


fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(6,4))
ax = axes.flat[0]

Z = np.zeros((20,5000))  
colors = ['k--', 'r--', 'g--']
labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']

for wv in range(0,3):            
    for ntrial in range(0,5000):
        for nAmount in range(0,20):
            Z[nAmount,ntrial]=trial_sat_pressure[ntrial,wv+9,nAmount]

    freq, bins = np.histogram(Z, 40, range=[-1000.,1000.], density=True)
#    freq = [(freq[i]/np.sum(freq)) for i in range(len(freq))]
    ax.plot(bins[:-1], freq*2000/40,  colors[wv], label=labels[wv])
ax.set_xlabel(r'$\delta$ P (CTP Bias)', fontsize=12)
ax.set_ylabel('Normalised number of occurences', fontsize=13)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.legend(fontsize=12)
#

ax = axes.flat[1]
Z = np.zeros((20,5000))
for wv in range(0,3):            
    for ntrial in range(0,5000):
        for nAmount in range(0,20):
            Z[nAmount,ntrial]=trial_sat_pressure[ntrial,wv+15,nAmount]
    meanAverage = np.mean(Z)
    standardDeviation = np.std(Z)
    weights = np.ones_like(Z)/float(len(Z))
    freq, bins = np.histogram(Z, 10, range=[-0.5,0.5], density=True)
#    freq = [(freq[i]-np.min(freq))/(np.max(freq)-np.min(freq))
#             for i in range(len(freq))]
    ax.plot(bins[:-1], freq*1/10,  colors[wv], label=labels[wv])
plt.xlabel(r'$\delta (\epsilon A_{c})$ (Cloud Amount Bias)', fontsize=12)
#plt.ylabel('Probability', fontsize=13)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.legend(fontsize=12)
fig.subplots_adjust(right=0.9)

plt.savefig('./plots/MODIS_CO2bands_errors_hist.png', dpi=200)
plt.show()
#


bias = np.zeros((25,21,3))
c = np.zeros((25,21,3))
dP = (1000-0.25)/25
colors = ['k--', 'r--', 'g--']
labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
for wv in range(0,3):   
    for nlevel in range(0,24):
        start_pres = 0.2 + (1000-0.25)/25 * nlevel 
        for nAmount in range(0,20):  
            indices = np.where(np.logical_and(trial_sat_pressure[:,4,nAmount]>start_pres,
                                              trial_sat_pressure[:,4,nAmount]<=start_pres+dP))
#            print(indices[0][:])
            c[nlevel,nAmount] = len(indices[0][:])
            for i in range(0, np.size(indices)):
                   if (np.isnan(trial_sat_pressure[indices[0][i],wv+6,nAmount])==False):
                       bias[nlevel,nAmount,wv] = bias[nlevel,nAmount,wv] + \
                       (trial_sat_pressure[indices[0][i],wv+6,nAmount]- \
                              trial_sat_pressure[indices[0][i],4,nAmount])
                   else:
                       c[nlevel,nAmount]=c[nlevel,nAmount]-1
    bias[:,:,wv] = np.divide(bias[:,:,wv], c[:,:,wv]+1, out=np.zeros_like(bias[:,:,wv]),
       where=c[:,:,wv]!=0)
###plt.figure(figsize=(8,6))
###ax=plt.gca()        
###XX=np.arange(0,20)
###YY=np.arange(0,25)
###X,Y=np.meshgrid(XX,YY)
###im=ax.contour(X,Y,Z,origin='lower',cmap='seismic')
###plt.colorbar(im)
###plt.show()
##
fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(16,10))
for i in range(0,3):
    ax=axes.flat[i]   
    im=ax.imshow(bias[:,:,i], cmap='RdBu_r', vmin=-500., vmax=500.)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = np.arange(-200,1200,200)
    labels = ax.set_yticklabels(labels)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [-0.125, 0.05, '', 0.5, '', 1.]
    labels = ax.set_xticklabels(labels)
    if (i==1):
        ax.set_xlabel('Effective Cloud Amount ($\epsilon A_{c}$)', fontsize=15)
    if (i==0):
        ax.set_ylabel('CTP (hPa)', fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
cb=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cb.set_label(r'Bias errors in CTP (hPa)', fontsize=13)
plt.savefig('./plots/MODIS_CO2bands_bias_ctp.png', dpi=400, bbox_to_inches='tight')
plt.show()
#
#
rms = np.zeros((25,21,3))
c = np.zeros((25,21,3))
dP = (1000-0.25)/25
colors = ['k--', 'r--', 'g--']
labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
for wv in range(0,3):   
    for nlevel in range(0,24):
        start_pres = 0.2 + (1000-0.25)/25 * nlevel 
        for nAmount in range(0,20):  
            indices = np.where(np.logical_and(trial_sat_pressure[:,4,nAmount]>start_pres,
                                              trial_sat_pressure[:,4,nAmount]<=start_pres+dP))
#            print(indices[0][:])
            c[nlevel,nAmount] = len(indices[0][:])
            for i in range(0, np.size(indices)):
#                   print(trial_sat_pressure[indices[0][i],6,nAmount],
#                        trial_sat_pressure[indices[0][i],4,nAmount])
#                   print(rms[nlevel,nAmount,wv] + (trial_sat_pressure[indices[0][:],wv+6,nAmount]- \
#                          trial_sat_pressure[indices[0][:],4,nAmount])^2)
                   if (np.isnan(trial_sat_pressure[indices[0][i],wv+6,nAmount])==False):
                       rms[nlevel,nAmount,wv] = rms[nlevel,nAmount,wv] + \
                       (trial_sat_pressure[indices[0][i],wv+6,nAmount]- \
                              trial_sat_pressure[indices[0][i],4,nAmount])**2
                   else:
                       c[nlevel,nAmount]=c[nlevel,nAmount]-1
    rms[:,:,wv] = np.sqrt(np.divide(rms[:,:,wv], c[:,:,wv]+1, out=np.zeros_like(rms[:,:,wv]),
       where=c[:,:,wv]!=0))
    
fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(16,10))
for i in range(0,3):
    ax=axes.flat[i]        
    im = ax.imshow(rms[:,:,i], cmap='Reds', vmin=0.05, vmax=500.)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = np.arange(-200,1200,200)
    labels = ax.set_yticklabels(labels)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [-0.125, 0.05, '', 0.5, '', 1.]
    labels = ax.set_xticklabels(labels)
    if (i==1):
        ax.set_xlabel('Effective Cloud Amount ($\epsilon A_{c}$)', fontsize=15)
    if (i==0):
        ax.set_ylabel('CTP (hPa)', fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
cb=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cb.set_label(r'RMS errors in CTP (hPa)', fontsize=13)
plt.savefig('./plots/MODIS_CO2bands_rmse_ctp.png', dpi=400, bbox_to_inches='tight')
plt.show()       