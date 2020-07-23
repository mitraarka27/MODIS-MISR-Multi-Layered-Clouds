# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:27:34 2020

@author: arkam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from scipy.stats import norm

global h
hn = 6.626e-34
c = 299792458
kB = 1.38e-23

lookup = np.loadtxt('MODIS_StdUSAtm_props.txt')
Pres=lookup[:,0]
Temp=lookup[:,1]
fT = interp1d(Pres[::10],Temp[::10])
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
intensity_at_level=np.zeros((len(Pres)-40,10))
ratio_at_level=np.zeros((len(Pres)-40,4,10))
intensity_at_level[:,4]=[Pres[j] for j in range(20,len(Pres)-20)]
trial_pressure=np.zeros((60,1000,10))
trial_ratio=np.zeros((60,1000,6))
band_snr=[0.75,1.0,1.0,1.25]
eff_am=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#Now let's simulate the satellite-observed TOA radiances
sat_radiances=np.zeros((len(Pres)-40,6,10))
sat_ratio=np.zeros((len(Pres)-40,5,10))
trial_sat_pressure=np.zeros((60,1000,24,10))
trial_sat_ratio=np.zeros((60,1000,3,10))

def planckfn(wavelength, T2):
    #Function to calculate Planck radiation for emissivity=1, given wavelength and Temp
    a = 2.0*hn*c**2
    b = hn*c/(wavelength*kB*T2)
    intensity = a/((wavelength**5)*(np.exp(b)-1.0))
    return intensity*1e-6

def trapezoidal(f, n):
    s = 0.0
    s += f[0]/2.0
    for i in range(1, n-1):
        s += f[i]
    s += f[n-1]/2.0
    return s * h

def I_integral(B,wt_fn,wv_n):
    return(B[:,wv_n]*wt_fn[:,wv_n])
    
wav = [13.335*1e-6,13.635*1e-6,13.935*1e-6]
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
        for nAmount in range(0,10):
            sat_radiances[j-20,wv,nAmount] = Ics[wv]+eff_am[nAmount]* \
                (intensity_at_level[j-20,wv]-Ics[wv])
            sat_radiances[j-20,5,nAmount] = eff_am[nAmount]
            

#For black clouds (nAmount=20, e=1) and non-black clouds 
#Calculated radiance ratios at known pressure levels
for wv in range(0,3):
    for j in range (20,len(Pres)-20):
        for nAmount in range(0,9):
            ratio_at_level[j-20,3,nAmount] = Pres[j]
            ratio_at_level[j-20,wv,nAmount] = (sat_radiances[j-\
                          20,wv+1,nAmount]-Ics[wv+1])/(sat_radiances[j-20,wv,nAmount]-Ics[wv])    

randp = np.linspace(100.,700.,61) #Specified upper CTP

f1 = interp1d(sat_radiances[:,4,9],sat_radiances[:,0,9])

f2 = interp1d(sat_radiances[:,4,9],sat_radiances[:,1,9])

f3 = interp1d(sat_radiances[:,4,9],sat_radiances[:,2,9])

f4 = interp1d(sat_radiances[:,4,9],sat_radiances[:,3,9])


for i in range(0,len(randp)-1):
    trial_sat_pressure[i,:,8,0] = randp[i]
    trial_sat_pressure[i,:,8,1] = randp[i]
    trial_sat_pressure[i,:,8,2] = randp[i]
    trial_sat_pressure[i,:,8,3] = randp[i]
    trial_sat_pressure[i,:,8,4] = randp[i]
    trial_sat_pressure[i,:,8,5] = randp[i]
    trial_sat_pressure[i,:,8,6] = randp[i]
    trial_sat_pressure[i,:,8,7] = randp[i]
    trial_sat_pressure[i,:,8,8] = randp[i]
    trial_sat_pressure[i,:,8,9] = randp[i]
    randlow = randp[i] + \
           np.random.rand(1000)*(np.max(sat_radiances[:,4,4])-\
                          randp[i] - 70.) #Random lower CTP 
    trial_sat_pressure[i,:,10,0] = randlow
    trial_sat_pressure[i,:,10,1] = randlow
    trial_sat_pressure[i,:,10,2] = randlow
    trial_sat_pressure[i,:,10,3] = randlow
    trial_sat_pressure[i,:,10,4] = randlow
    trial_sat_pressure[i,:,10,5] = randlow
    trial_sat_pressure[i,:,10,6] = randlow
    trial_sat_pressure[i,:,10,7] = randlow
    trial_sat_pressure[i,:,10,8] = randlow
    trial_sat_pressure[i,:,10,9] = randlow
    

    #Approximate cloudy radiance at trial upper & lower pressure levels
    trial_sat_pressure[i,:,0,9]=f1(trial_sat_pressure[i,:,8,9])
    trial_sat_pressure[i,:,1,9]=f2(trial_sat_pressure[i,:,8,9])
    trial_sat_pressure[i,:,2,9]=f3(trial_sat_pressure[i,:,8,9])
    trial_sat_pressure[i,:,3,9]=f4(trial_sat_pressure[i,:,8,9])
    trial_sat_pressure[i,:,4,9]=f1(trial_sat_pressure[i,:,10,9])
    trial_sat_pressure[i,:,5,9]=f2(trial_sat_pressure[i,:,10,9])
    trial_sat_pressure[i,:,6,9]=f3(trial_sat_pressure[i,:,10,9])
    trial_sat_pressure[i,:,7,9]=f4(trial_sat_pressure[i,:,10,9])

    for nAmount in range(0,9):
        trial_sat_pressure[i,:,9,nAmount]=eff_am[nAmount]
        trial_sat_pressure[i,:,0,nAmount]=Ics[wv]+(1-eff_am[nAmount])*(trial_sat_pressure[i,:,4,9]-
                                            Ics[wv])+eff_am[nAmount]*(trial_sat_pressure[i,:,0,9]-Ics[wv])
        trial_sat_pressure[i,:,1,nAmount]=Ics[wv]+(1-eff_am[nAmount])*(trial_sat_pressure[i,:,5,9]-
                                            Ics[wv])+eff_am[nAmount]*(trial_sat_pressure[i,:,1,9]-Ics[wv])
        trial_sat_pressure[i,:,2,nAmount]=Ics[wv]+(1-eff_am[nAmount])*(trial_sat_pressure[i,:,6,9]-
                                            Ics[wv])+eff_am[nAmount]*(trial_sat_pressure[i,:,2,9]-Ics[wv])
        trial_sat_pressure[i,:,3,nAmount]=Ics[wv]+(1-eff_am[nAmount])*(trial_sat_pressure[i,:,7,9]-
                                            Ics[wv])+eff_am[nAmount]*(trial_sat_pressure[i,:,3,9]-Ics[wv])

#Let's add Gaussian noise to satellite signals
# noise = np.random.normal(0, 0.25, (50,1000))
# trial_sat_pressure[:,:,0,nAmount]=trial_sat_pressure[i,:,0,nAmount]+noise
# noise = np.random.normal(0, 0.25, (50,1000))
# trial_sat_pressure[:,:,1,nAmount]=trial_sat_pressure[i,:,1,nAmount]+noise
# noise = np.random.normal(0, 0.25, (50,1000))
# trial_sat_pressure[:,:,2,nAmount]=trial_sat_pressure[i,:,2,nAmount]+noise
# noise = np.random.normal(0, 0.25, (50,1000))
# trial_sat_pressure[:,:,3,nAmount]=trial_sat_pressure[i,:,3,nAmount]+noise
            

for ntrial in range(0,60):
    for lcld in range(0,1000):
        for wv in range(0,3):
            for nAmount in range(0,9):
                #Observed radiance ratio
                # if (trial_sat_pressure[ntrial,lcld,wv+1,nAmount]- Ics[wv+1]) < band_snr[wv+1] and \
                #     (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv]) < band_snr[wv]:
                        trial_sat_ratio[ntrial,lcld,wv,nAmount]=(trial_sat_pressure[ntrial,lcld,wv+1,nAmount]-\
                                   Ics[wv+1])/(trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv]) 

for ntrial in range(0,60):
    for lcld in range(0,1000):    
        for wv in range(0,3):
            for nAmount in range(0,9):
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
                    if wv == 0:
                        trial_sat_pressure[ntrial,lcld,wv+17,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv])/ \
                            (f1(trial_sat_pressure[ntrial,lcld,wv+11,nAmount])-Ics[wv]) 
                    elif wv == 1:
                        trial_sat_pressure[ntrial,lcld,wv+17,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv])/ \
                            (f2(trial_sat_pressure[ntrial,lcld,wv+11,nAmount])-Ics[wv]) 
                    else:
                        trial_sat_pressure[ntrial,lcld,wv+17,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv,nAmount]-Ics[wv])/ \
                            (f3(trial_sat_pressure[ntrial,lcld,wv+11,nAmount])-Ics[wv]) 
                    # print(trial_sat_pressure[ntrial,lcld,wv,nAmount],Ics[wv],B_CTP)
                    trial_sat_pressure[ntrial,lcld,wv+20,nAmount]= \
                        (trial_sat_pressure[ntrial,lcld,wv+17,nAmount]- \
                         trial_sat_pressure[ntrial,lcld,9,nAmount])
                    if trial_sat_pressure[ntrial,lcld,wv+17,nAmount]>1 or \
                       trial_sat_pressure[ntrial,lcld,wv+17,nAmount]<0. :
                        trial_sat_pressure[ntrial,lcld,wv+17,nAmount]=np.nan
                        trial_sat_pressure[ntrial,lcld,wv+20,nAmount]=np.nan
                except:
                    trial_sat_pressure[ntrial,lcld,wv+11,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+14,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+17,nAmount]=np.nan
                    trial_sat_pressure[ntrial,lcld,wv+20,nAmount]=np.nan

#------------------------------------------Plot1--------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# errorbylow = np.zeros((60,9,3,25))
# c = np.zeros((60,9,3,25))
# # dP = (1000-0.25)/25
# # top_p = [0., 200., 400.]
# # bottom_p = [200., 400., 600.]
# #colors = ['k--', 'r--', 'g--']
# labels = ['$13.3 \mu m$', '$13.6 \mu m$', '$13.9 \mu m$']
# for i in range(0,60):
#     dP =  (1000.-trial_sat_pressure[i,0,8,0])/25.
#     for wv in range(0,3):
#         for nAmount in range(0,9,2):
#             for r in range(0,25):
#                 start_pres = trial_sat_pressure[i,0,8,0] + dP * r
#                 indices = np.where(np.logical_and(trial_sat_pressure[i,:,10,nAmount]>start_pres,
#                                                   trial_sat_pressure[i,:,10,nAmount]<=start_pres+dP))
#                 c[i,nAmount,wv,r] = c[i,nAmount,wv,r]+len(indices[0][:])
#                 for j in range(0, np.size(indices)):
#                     if (np.isnan(trial_sat_pressure[i,indices[0][j],wv+20,nAmount])==False):
#                                   errorbylow[i,nAmount,wv,r] = errorbylow[i,nAmount,wv,r] + \
#                                       trial_sat_pressure[i,indices[0][j],wv+20,nAmount]

#                     else:
#                                   c[i,nAmount,wv,r]=c[i,nAmount,wv,r]-1
#             errorbylow[i,nAmount,wv,:] = np.divide(errorbylow[i,nAmount,wv,:], c[i,nAmount,wv,:]+1, 
#                                                         out=np.zeros_like(errorbylow[i,nAmount,wv,:]), where=c[i,nAmount,wv,:]!=0)

# P = np.linspace(400,1000, 25)
# errorbylow[np.where(errorbylow==0)]=np.nan
# fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(10,6))
# for i in range(0,3):
#     ax=axes.flat[i] 
#     ax.plot(P[:-1],errorbylow[30,0,i,:-1],label='$\epsilon A_c = $'+str(eff_am[0]));
#     ax.plot(P[:-1],errorbylow[30,2,i,:-1],label='$\epsilon A_c = $'+str(eff_am[2]));
#     ax.plot(P[:-1],errorbylow[30,4,i,:-1],label='$\epsilon A_c = $'+str(eff_am[4]));
#     ax.plot(P[:-1],errorbylow[30,6,i,:-1],label='$\epsilon A_c = $'+str(eff_am[6]));
#     ax.plot(P[:-1],errorbylow[30,8,i,:-1],label='$\epsilon A_c = $'+str(eff_am[8]));
#     ax.set_xlim(0,1000.);ax.set_ylim(0,1.0);
#     ax.set_title(labels[i], fontsize=13)
#     if (i==2):
#         ax.legend(fontsize=13, bbox_to_anchor=(0.7, 1.05), fancybox=True, shadow=True)
#     if (i==1):
#         ax.set_xlabel('CTP for lower cloud (hPa)', fontsize=13);
#     if (i==0):
#         ax.set_ylabel('Overestimation of top-layer Effective \n Amount ($\epsilon A_c$) for low CTP = 400 hPa', fontsize=13);
# plt.savefig('./plots/MODIS_CO2bands_2layers_bias_bylowCTP_effAmt.png', dpi=400, bbox_to_inches='tight')
# plt.show()

# #------------------------------------------Plot2--------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

# errorbylow = np.zeros((60,9,3,25))
# c = np.zeros((60,9,3,25))
# # dP = (1000-0.25)/25
# # top_p = [0., 200., 400.]
# # bottom_p = [200., 400., 600.]
# #colors = ['k--', 'r--', 'g--']
# labels = ['$R_{1/2}$', '$R_{2/3}$', '$R_{3/4}$']
# for i in range(0,60):
#     dP =  (1000.-trial_sat_pressure[i,0,8,0])/25.
#     for wv in range(0,3):
#         for nAmount in range(0,9,2):
#             for r in range(0,25):
#                 start_pres = trial_sat_pressure[i,0,8,0] + dP * r
#                 indices = np.where(np.logical_and(trial_sat_pressure[i,:,10,nAmount]>start_pres,
#                                                   trial_sat_pressure[i,:,10,nAmount]<=start_pres+dP))
#                 c[i,nAmount,wv,r] = c[i,nAmount,wv,r]+len(indices[0][:])
#                 for j in range(0, np.size(indices)):
#                     if (np.isnan(trial_sat_pressure[i,indices[0][j],wv+20,nAmount])==False):
#                                   errorbylow[i,nAmount,wv,r] = errorbylow[i,nAmount,wv,r] + \
#                             (trial_sat_pressure[i,indices[0][j],wv+11,nAmount]- \
#                                   trial_sat_pressure[i,indices[0][j],8,nAmount])

#                     else:
#                                   c[i,nAmount,wv,r]=c[i,nAmount,wv,r]-1
#             errorbylow[i,nAmount,wv,:] = np.divide(errorbylow[i,nAmount,wv,:], c[i,nAmount,wv,:]+1, 
#                                                         out=np.zeros_like(errorbylow[i,nAmount,wv,:]), where=c[i,nAmount,wv,:]!=0)

# P = np.linspace(400,1000, 25)
# errorbylow[np.where(errorbylow==0)]=np.nan
# fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(10,6))
# for i in range(0,3):
#     ax=axes.flat[i] 
#     ax.plot(P,errorbylow[30,0,i,:],label='$\epsilon A_c = $'+str(eff_am[0]));
#     ax.plot(P,errorbylow[30,2,i,:],label='$\epsilon A_c = $'+str(eff_am[2]));
#     ax.plot(P,errorbylow[30,4,i,:],label='$\epsilon A_c = $'+str(eff_am[4]));
#     ax.plot(P,errorbylow[30,6,i,:],label='$\epsilon A_c = $'+str(eff_am[6]));
#     ax.plot(P,errorbylow[30,8,i,:],label='$\epsilon A_c = $'+str(eff_am[8]));
#     ax.set_xlim(0,1000.);ax.set_ylim(0,300.);
#     ax.set_title(labels[i], fontsize=13)
#     if (i==2):
#         ax.legend(fontsize=13, bbox_to_anchor=(1.05, 1.0), fancybox=True, shadow=True)
#     if (i==1):
#         ax.set_xlabel('CTP for lower cloud (hPa)', fontsize=13);
#     if (i==0):
#         ax.set_ylabel('Overestimation of top-layer Cloud-Top \n Pressure (CTP) for low CTP = 400 hPa', fontsize=13);
# plt.savefig('./plots/MODIS_CO2bands_2layers_bias_bylowCTP_CTP.png', dpi=400, bbox_to_inches='tight')
# plt.show()

# #------------------------------------------Regression--------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
                    
# Create master dataset for multi-variate linear regression
master = np.zeros((3,60*1000*9+1000*9+9,6))
for wv in range(0,3):
    for ntrial in range(0,60):
        for lcld in range(0,1000):    
            for nAmount in range(0,9):
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,0]=\
                trial_sat_pressure[ntrial,lcld,10,nAmount]
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,1]=\
                trial_sat_pressure[ntrial,lcld,17,nAmount]
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,2]=\
                trial_sat_pressure[ntrial,lcld,18,nAmount]
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,3]=\
                trial_sat_pressure[ntrial,lcld,19,nAmount]
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,4]=\
                trial_sat_pressure[ntrial,lcld,wv+11,nAmount]
                master[wv,ntrial*lcld*nAmount+lcld*nAmount+nAmount,5]=\
                trial_sat_pressure[ntrial,lcld,wv+20,nAmount]

master2 = master[0,:,:]
master2 = master2[~np.isnan(master2).any(axis=1)]
X = master2[:,:4]
Y = master2[:,-1]

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

coeff_df = pd.DataFrame(model.coef_, ['Lcld_P','W1_em','W2_em','W3_em'], columns=['Coefficient'])
print(coeff_df)
y_pred = model.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
print(df.head(100))

# Root Mean Squared Deviation
rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))      
r2_value = r2_score(Y_test, y_pred)                     

print("Intercept: \n", model.intercept_)
print("Root Mean Square Error \n", rmsd)
print("R^2 Value: \n", r2_value)