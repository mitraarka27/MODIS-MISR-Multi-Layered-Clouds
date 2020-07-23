# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:12:06 2020

@author: arkam
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
ratio_at_level=np.zeros((len(Pres)-40,4,5))
intensity_at_level[:,4]=[Pres[j] for j in range(20,len(Pres)-20)]
trial_pressure=np.zeros((5000,1000,10))
trial_ratio=np.zeros((5000,1000,6))
band_snr=[0.75,1.0,1.0,1.25]
eff_am=[0.10, 0.25, 0.50, 0.75, 1.00]

#Now let's simulate the satellite-observed TOA radiances
sat_radiances=np.zeros((len(Pres)-40,6,5))
sat_ratio=np.zeros((len(Pres)-40,5,5))
trial_sat_pressure=np.zeros((5000,1000,24,5))
trial_sat_ratio=np.zeros((5000,1000,3,5))

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
        for nAmount in range(0,5):
            sat_radiances[j-20,wv,nAmount] = Ics[wv]+eff_am[nAmount]* \
                (intensity_at_level[j-20,wv]-Ics[wv])
            sat_radiances[j-20,5,nAmount] = eff_am[nAmount]
            

#For black clouds (nAmount=20, e=1) and non-black clouds 
#Calculated radiance ratios at known pressure levels
for wv in range(0,3):
    for j in range (20,len(Pres)-20):
        for nAmount in range(0,4):
            ratio_at_level[j-20,3,nAmount] = Pres[j]
            ratio_at_level[j-20,wv,nAmount] = (sat_radiances[j-\
                          20,wv+1,nAmount]-Ics[wv+1])/(sat_radiances[j-20,wv,nAmount]-Ics[wv])    

randp = np.min(sat_radiances[:,4,4])+ \
           np.random.rand(5000)*(np.max(sat_radiances[:,4,4])-\
                          np.min(sat_radiances[:,4,4])) #Random upper CTP

f1 = interp1d(sat_radiances[:,4,4],sat_radiances[:,0,4])

f2 = interp1d(sat_radiances[:,4,4],sat_radiances[:,1,4])

f3 = interp1d(sat_radiances[:,4,4],sat_radiances[:,2,4])

f4 = interp1d(sat_radiances[:,4,4],sat_radiances[:,3,4])


for i in range(0,len(randp)):
    trial_sat_pressure[i,:,8,0] = randp[i]
    trial_sat_pressure[i,:,8,1] = randp[i]
    trial_sat_pressure[i,:,8,2] = randp[i]
    trial_sat_pressure[i,:,8,3] = randp[i]
    trial_sat_pressure[i,:,8,4] = randp[i]
    randlow = randp[i] + \
           np.random.rand(1000)*(np.max(sat_radiances[:,4,4])-\
                          randp[i]) #Random lower CTP 
    trial_sat_pressure[i,:,10,0] = randlow
    trial_sat_pressure[i,:,10,1] = randlow
    trial_sat_pressure[i,:,10,2] = randlow
    trial_sat_pressure[i,:,10,3] = randlow
    trial_sat_pressure[i,:,10,4] = randlow
    

    #Approximate cloudy radiance at trial upper & lower pressure levels
    trial_sat_pressure[i,:,0,4]=f1(trial_sat_pressure[i,:,8,4])
    trial_sat_pressure[i,:,1,4]=f2(trial_sat_pressure[i,:,8,4])
    trial_sat_pressure[i,:,2,4]=f3(trial_sat_pressure[i,:,8,4])
    trial_sat_pressure[i,:,3,4]=f4(trial_sat_pressure[i,:,8,4])
    trial_sat_pressure[i,:,4,4]=f1(trial_sat_pressure[i,:,10,4])
    trial_sat_pressure[i,:,5,4]=f2(trial_sat_pressure[i,:,10,4])
    trial_sat_pressure[i,:,6,4]=f3(trial_sat_pressure[i,:,10,4])
    trial_sat_pressure[i,:,7,4]=f4(trial_sat_pressure[i,:,10,4])

    for nAmount in range(0,len(eff_am)):
        trial_sat_pressure[i,:,9,nAmount]=eff_am[nAmount]
        
        f11 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,0,nAmount])
        
        f22 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,1,nAmount])
        
        f33 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,2,nAmount])
        
        f44 = interp1d(sat_radiances[:,4,nAmount],sat_radiances[:,3,nAmount])
    
        #Simulated satellite observed radiances at each trial pressure level for
        #given values of cloud emissivities, in non-black cases.
        trial_sat_pressure[i,:,0,nAmount]=f11(trial_sat_pressure[i,:,8,nAmount])
        trial_sat_pressure[i,:,1,nAmount]=f22(trial_sat_pressure[i,:,8,nAmount])
        trial_sat_pressure[i,:,2,nAmount]=f33(trial_sat_pressure[i,:,8,nAmount])
        trial_sat_pressure[i,:,3,nAmount]=f44(trial_sat_pressure[i,:,8,nAmount])
        #Only considering completely emissive lower clouds at the moment
        trial_sat_pressure[i,:,4,nAmount]=f11(trial_sat_pressure[i,:,10,4])
        trial_sat_pressure[i,:,5,nAmount]=f22(trial_sat_pressure[i,:,10,4])
        trial_sat_pressure[i,:,6,nAmount]=f33(trial_sat_pressure[i,:,10,4])
        trial_sat_pressure[i,:,7,nAmount]=f44(trial_sat_pressure[i,:,10,4])
        #Adjustingfor the radiance between the lower and upper clouds
        trial_sat_pressure[i,:,0,nAmount]=trial_sat_pressure[i,:,0,nAmount]+ \
                                    (1.0-eff_am[nAmount])*(Ics[0]-trial_sat_pressure[i,:,4,nAmount])
        trial_sat_pressure[i,:,1,nAmount]=trial_sat_pressure[i,:,1,nAmount]+ \
                                    (1.0-eff_am[nAmount])*(Ics[1]-trial_sat_pressure[i,:,5,nAmount])
        trial_sat_pressure[i,:,2,nAmount]=trial_sat_pressure[i,:,2,nAmount]+ \
                                    (1.0-eff_am[nAmount])*(Ics[2]-trial_sat_pressure[i,:,6,nAmount])
        trial_sat_pressure[i,:,3,nAmount]=trial_sat_pressure[i,:,3,nAmount]+ \
                                    (1.0-eff_am[nAmount])*(Ics[3]-trial_sat_pressure[i,:,7,nAmount])

#Let's add Gaussian noise to satellite signals
noise = np.random.normal(0, 0.25, (5000,1000))
trial_sat_pressure[:,:,0,nAmount]=trial_sat_pressure[i,:,0,nAmount]+noise
noise = np.random.normal(0, 0.25, (5000,1000))
trial_sat_pressure[:,:,1,nAmount]=trial_sat_pressure[i,:,1,nAmount]+noise
noise = np.random.normal(0, 0.25, (5000,1000))
trial_sat_pressure[:,:,2,nAmount]=trial_sat_pressure[i,:,2,nAmount]+noise
noise = np.random.normal(0, 0.25, (5000,1000))
trial_sat_pressure[:,:,3,nAmount]=trial_sat_pressure[i,:,3,nAmount]+noise
            

for ntrial in range(0,5000):
    for lcld in range(0,1000):
        for wv in range(0,3):
            for nAmount in range(0,5):
                #Observed radiance ratio
                if (trial_sat_pressure[ntrial,lcld,wv+1,nAmount]- Ics[wv+1]) < band_snr[wv+1] and \
                    (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv]) < band_snr[wv]:
                        trial_sat_ratio[ntrial,lcld,wv,nAmount]=(trial_sat_pressure[ntrial,lcld,wv+1,nAmount]-\
                                   Ics[wv+1])/(trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv]) 

for ntrial in range(0,5000):
    for lcld in range(0,1000):    
        for wv in range(0,3):
            for nAmount in range(0,5):
                z = trial_sat_ratio[ntrial,lcld,wv,nAmount]-ratio_at_level[:,wv,nAmount]
                try:
                    asign = np.sign(z)
                    signchange = (np.abs((np.roll(asign, 1) - asign)) != 0).astype(int)
                    signchange[0]=0
                    obs_pres=sat_radiances[np.min(np.where(signchange==1)),4,nAmount]
                    if obs_pres>900. or obs_pres<100.:
                        obs_pres=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+11,nAmount]=obs_pres
                    trial_sat_pressure[ntrial,lcld,wv+14,nAmount]= \
                        obs_pres-trial_sat_pressure[ntrial,lcld,8,nAmount]
                    trial_sat_pressure[ntrial,lcld,wv+17,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv])/ \
                        (trial_sat_pressure[ntrial,lcld,wv,4]-Ics[wv])
                    trial_sat_pressure[ntrial,lcld,wv+20,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv+17,nAmount]- \
                         trial_sat_pressure[ntrial,lcld,9,nAmount])
                except:
                    trial_sat_pressure[ntrial,lcld,wv+11,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+14,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+17,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+20,nAmount]=np.nan

#------------------------------------------Plot1--------------------------------------------------------
#-------------------------------------------------------------------------------------------------------


# bias = np.zeros((25,25,5,3))
# c = np.zeros((25,25,5,3))
# #dP = (1000-0.25)/25
# #colors = ['k--', 'r--', 'g--']
# wvlabels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
# for wv in range(0,3):
#     for nAmount in range(0,5):   
#         for nlevel in range(0,5000):
#             m = int((trial_sat_pressure[nlevel,0,8,nAmount]-0.20)*25/1000.) 
#             for lcld in range(0,1000):
#                 n = int((trial_sat_pressure[nlevel,lcld,10,nAmount]-0.20)*25/1000)
                
#                 if (np.isnan(trial_sat_pressure[nlevel,lcld,wv+11,nAmount])==False):
#                            bias[m,n,nAmount,wv] = bias[m,n,nAmount,wv] + \
#                            (trial_sat_pressure[m,n,wv+11,nAmount]- \
#                                   trial_sat_pressure[m,n,8,nAmount])
#                            c[m,n,nAmount,wv] = c[m,n,nAmount,wv] + 1

#         bias[:,:,nAmount,wv] = np.divide(bias[:,:,nAmount,wv], c[:,:,nAmount,wv]+1, \
#             out=np.zeros_like(bias[:,:,nAmount, wv]), where=c[:,:,nAmount, wv]!=0)

# ##
# fig, axes = plt.subplots(nrows = 3, ncols=4, figsize=(16,10))
# for wv in range(0,3):
#     for i in range(0,4):
#         ax=axes.flat[4*wv+i]   
#         im=ax.imshow(bias[:,:,i,wv], cmap='seismic', vmin=-400., vmax=400.)
#         labels = [item.get_text() for item in ax.get_yticklabels()]
#         labels = np.arange(-200,1200,200)
#         labels = ax.set_yticklabels(labels)
#     #    labels = [item.get_text() for item in ax.get_xticklabels()]
#         ax.invert_xaxis()
#         labels = np.arange(-200,1200,200)
#         labels = ax.set_xticklabels(labels)
#         if (wv==2 and i==2):
#             ax.set_xlabel('Cloud Top Pressure for the Lower Cloud (hPa)', fontsize=15)
#         if (wv!=1 and i==0):
#             ax.set_ylabel(str(wvlabels[wv]), fontsize=15)
#         if (wv==1 and i==0):
#             ax.set_ylabel('CTP (hPa) \n '+str(wvlabels[1]), fontsize=15)
#             for tick in ax.yaxis.get_major_ticks():
#                 tick.label.set_fontsize(12)
#         for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(12)
#         if(wv==0):
#             ax.set_title(r'$\epsilon A_{c}$ = '+str(eff_am[i]), fontsize=14)
# cb=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
# cb.set_label(r'Bias errors in CTP (hPa)', fontsize=13)
# plt.savefig('./plots/MODIS_CO2bands_2layers_ratio1_bias_ctp.png', dpi=400, bbox_to_inches='tight')
# plt.show()

# #------------------------------------------Plot2--------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

# rmse = np.zeros((25,25,5,3))
# c = np.zeros((25,25,5,3))
# #dP = (1000-0.25)/25
# #colors = ['k--', 'r--', 'g--']
# #labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
# for wv in range(0,3):
#     for nAmount in range(0,5):   
#         for nlevel in range(0,5000):
#             m = int((trial_sat_pressure[nlevel,0,8,nAmount]-0.20)*25/1000.) 
#             for lcld in range(0,1000):
#                 n = int((trial_sat_pressure[nlevel,lcld,10,nAmount]-0.20)*25/1000)
                
#                 if (np.isnan(trial_sat_pressure[nlevel,lcld,wv+11,nAmount])==False):
#                        rmse[m,n,nAmount,wv] = rmse[m,n,nAmount,wv] + \
#                        (trial_sat_pressure[m,n,wv+11,nAmount]- \
#                               trial_sat_pressure[m,n,8,nAmount])**2
#                        c[m,n,nAmount,wv] = c[m,n,nAmount,wv] + 1

#         rmse[:,:,nAmount,wv] = np.sqrt(np.divide(rmse[:,:,nAmount,wv], c[:,:,nAmount,wv]+1, \
#             out=np.zeros_like(rmse[:,:,nAmount, wv]), where=c[:,:,nAmount, wv]!=0))

# ##
# fig, axes = plt.subplots(nrows = 3, ncols=4, figsize=(16,10))
# for wv in range(0,3):
#     for i in range(0,4):
#         ax=axes.flat[4*wv+i]   
#         im=ax.imshow(rmse[:,:,i,wv], cmap='Reds', vmin=0., vmax=400.)
#         labels = [item.get_text() for item in ax.get_yticklabels()]
#         labels = np.arange(-200,1200,200)
#         labels = ax.set_yticklabels(labels)
#     #    labels = [item.get_text() for item in ax.get_xticklabels()]
#         ax.invert_xaxis()
#         labels = np.arange(-200,1200,200)
#         labels = ax.set_xticklabels(labels)
#         if (wv==2 and i==2):
#             ax.set_xlabel('Cloud Top Pressure for the Lower Cloud (hPa)', fontsize=15)
#         if (wv!=1 and i==0):
#             ax.set_ylabel(str(wvlabels[wv]), fontsize=15)
#         if (wv==1 and i==0):
#             ax.set_ylabel('CTP (hPa) \n '+str(wvlabels[1]), fontsize=15)
#             for tick in ax.yaxis.get_major_ticks():
#                 tick.label.set_fontsize(12)
#         for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(12)
#         if(wv==0):
#             ax.set_title(r'$\epsilon A_{c}$ = '+str(eff_am[i]), fontsize=14)
# cb=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
# cb.set_label(r'RMSE errors in CTP (hPa)', fontsize=13)
# plt.savefig('./plots/MODIS_CO2bands_2layers_ratio1_rmse_ctp.png', dpi=400, bbox_to_inches='tight')
# plt.show()

#------------------------------------------Plot3--------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

errorbylow = np.zeros((25,4,3))
c = np.zeros((25,4,3))
dP = (1000-0.25)/25
#colors = ['k--', 'r--', 'g--']
labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
for wv in range(0,3):
    for nAmount in range(0,4):
        for nlevel in range(0,24):
            start_pres = 0.2 + (1000-0.25)/25 * nlevel 
            for i in range(0,5000):
                indices = np.where(np.logical_and(trial_sat_pressure[i,:,10,nAmount]>start_pres,
                                                  trial_sat_pressure[i,:,10,nAmount]<=start_pres+dP))
                c[nlevel,nAmount,wv] = c[nlevel,nAmount,wv]+len(indices[0][:])
                for j in range(0, np.size(indices)):
                       if (np.isnan(trial_sat_pressure[i,indices[0][j],wv+11,nAmount])==False):
                           errorbylow[nlevel,nAmount,wv] = errorbylow[nlevel,nAmount,wv] + \
                           (trial_sat_pressure[i,indices[0][j],wv+11,nAmount]- \
                                  trial_sat_pressure[i,indices[0][j],8,nAmount])
                       else:
                           c[nlevel,nAmount,wv]=c[nlevel,nAmount,wv]-1
        errorbylow[:,nAmount,wv] = np.divide(errorbylow[:,nAmount,wv], c[:,nAmount,wv]+1, 
              out=np.zeros_like(errorbylow[:,nAmount,wv]), where=c[:,nAmount,wv]!=0)

P = np.linspace(0., 1000, 25)
fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(10,6))
for i in range(0,3):
    ax=axes.flat[i] 
    ax.plot(P,-errorbylow[:,0,i],label='$\epsilon A_c = $'+str(eff_am[0]));
    ax.plot(P,-errorbylow[:,1,i],label='$\epsilon A_c = $'+str(eff_am[1]));
    ax.plot(P,-errorbylow[:,2,i],label='$\epsilon A_c = $'+str(eff_am[2]));
    ax.plot(P,-errorbylow[:,3,i],label='$\epsilon A_c = $'+str(eff_am[3]));
    ax.set_xlim(0,1000);ax.set_ylim(0,300.);
    ax.set_title(labels[i], fontsize=13)
    if (i==2):
        ax.legend(fontsize=13);
    if (i==1):
        ax.set_xlabel('CTP for lower cloud (hPa)', fontsize=13);
    if (i==0):
        ax.set_ylabel('Overestimation of top layer \n cloud-top pressure (hPa)', fontsize=13);
plt.savefig('./plots/MODIS_CO2bands_2layers_bias_bylowCTP_ctp.png', dpi=400, bbox_to_inches='tight')
plt.show()

#------------------------------------------Plot4--------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

#errorbylow = np.zeros((25,4,3,3))
#c = np.zeros((25,4,3,3))
#dP = (1000-0.25)/25
#top_p = [0., 200., 400.]
#bottom_p = [200., 400., 600.]
##colors = ['k--', 'r--', 'g--']
#labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
#for wv in range(0,3):
#    for nAmount in range(0,4):
#        for nlevel in range(0,24):
#            start_pres = 0.2 + (1000-0.25)/25 * nlevel 
#            for r in range(0,3):
#                for i in range(0,5000):
#                    if np.logical_and(trial_sat_pressure[i,0,8,nAmount]>top_p[r],trial_sat_pressure[i,0,8,nAmount]<bottom_p[r]):
#                        indices = np.where(np.logical_and(trial_sat_pressure[i,:,10,nAmount]>start_pres,
#                                                          trial_sat_pressure[i,:,10,nAmount]<=start_pres+dP))
#                        c[nlevel,nAmount,wv,r] = c[nlevel,nAmount,wv,r]+len(indices[0][:])
#                        for j in range(0, np.size(indices)):
#                               if (np.isnan(trial_sat_pressure[i,indices[0][j],wv+17,nAmount])==False):
#                                   errorbylow[nlevel,nAmount,wv,r] = errorbylow[nlevel,nAmount,wv,r] + \
#                                       trial_sat_pressure[i,indices[0][j],9,nAmount]- \
#                                              trial_sat_pressure[i,indices[0][j],wv+17,nAmount]
#
#                               else:
#                                   c[nlevel,nAmount,wv,r]=c[nlevel,nAmount,wv,r]-1
#                errorbylow[:,nAmount,wv,r] = np.divide(errorbylow[:,nAmount,wv,r], c[:,nAmount,wv,r]+1, 
#                  out=np.zeros_like(errorbylow[:,nAmount,wv,r]), where=c[:,nAmount,wv,r]!=0)
#
#P = np.linspace(0., 1000, 25)
#fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(10,6))
#for i in range(0,3):
#    ax=axes.flat[i] 
#    ax.plot(P,errorbylow[:,0,i,0],label='$\epsilon A_c = $'+str(eff_am[0]));
#    ax.plot(P,errorbylow[:,1,i,0],label='$\epsilon A_c = $'+str(eff_am[1]));
#    ax.plot(P,errorbylow[:,2,i,0],label='$\epsilon A_c = $'+str(eff_am[2]));
#    ax.plot(P,errorbylow[:,3,i,0],label='$\epsilon A_c = $'+str(eff_am[3]));
#    ax.set_xlim(0,1000.);ax.set_ylim(0,1.0);
#    ax.set_title(labels[i], fontsize=13)
#    if (i==2):
#        ax.legend(fontsize=13);
#    if (i==1):
#        ax.set_xlabel('CTP for lower cloud (hPa)', fontsize=13);
#    if (i==0):
#        ax.set_ylabel('Overestimation of top layer \n Effective Cloud Amount ($\epsilon A_c$)', fontsize=13);
#plt.savefig('./plots/MODIS_CO2bands_2layers_bias_bylowCTP_effAmt.png', dpi=400, bbox_to_inches='tight')
#plt.show()

#file = open('./textfiles/CTPtests_2layered_1.txt', 'w')
#for i in range(0,5000):
#    for j in range(0,500):
#        for wv in range(0,3):
#            for nAmount in range(0,4):
#                file.write(str(trial_sat_pressure[i,j,8,0])+" "+str(trial_sat_pressure[i,j,10,0])+" "+\
#                      str(eff_am[nAmount])+" "+"1.00"+" "+\
#                      str(wv)+" "+str(trial_sat_pressure[i,j,wv+11,nAmount])+"\n")
#file.close()