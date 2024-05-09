#!/usr/bin/env python
# coding: utf-8
#author: Woodkensia Charles email: woodych6@gmail.com
# Code for senior of thesis of Time-Invariance Violation of Neutrino Oscillation
# # Table of Contents: 
# ### Imports
# ### Probability in a vacuum
#    #### Hamiltionian matrix
#    #### Evolution Matrix
#    #### Probability defined
#    #### Plotting
#    #### Info
#        
# ### Probability in constant vacuum
# 

# ### Info:<br>
# - Probability of electron neutrino oscillating into a muon neutrino (P_eu)
# (final_elec)U(initial_mu) <br>
# muon---->elec
# 
# - Probability of muon neutrino oscillating into a electron neutrino (P_ue)
# (final_mu)U(initial_elec) <br>
# elec---->muon
# 
# -  Probability of elec neutrino oscillating into a elec neutrino (P_ee)
# (initial_elec)U(initial elec)
# elec-----> elec
# 
# -  Probability of muon neutrino oscillating into a muon neutrino (P_uu)
# (initial_muon)U(initial muon)
# muon-----> muon
# 

# ## Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
import cmath
from math import log,exp,sin,cos,sqrt,pi,e,sqrt
from math import * #import every function from math
from numpy.linalg import inv
from numpy import *
from sympy import *
init_printing(use_unicode=True)
from numpy import linalg as LA
from numpy.linalg import eig
from numpy import zeros,empty,loadtxt
from matplotlib import cm
from matplotlib import *
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot,xlabel,ylabel,show
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']


# # Probability in a vacuum

# ### Hamiltionian matrix

# In[ ]:


def Hamiltionian_matrix(E,theta,deltamsq):
    #defining the hamilition matrix
    hamilt=np.array ([[np.sin(theta)**2, np.cos(theta)*np.sin(theta)],[np.cos(theta)*np.sin(theta),np.cos(theta)**2]])
    triangle= (deltamsq)/(2*E)
    hamilt=  hamilt*triangle
    
    
    return hamilt* 5.067730718


# In[ ]:


Hamiltionian_matrix(0.5,.58,10**-3)


# ### Evolution Matrix

# In[ ]:


#e^-i*L*eval

#define L_hamilitonian 
def Evolution_matrix(L,E,theta,deltamsq):

    
    hamilt= Hamiltionian_matrix(E,theta,deltamsq)
    
    #the eigenvalues(vals) and vectors (vec)
    eigen=LA.eig(hamilt) #so this function shows both vec and vals
    hamilteigenval= eigen[0] #this seperate to get the eigen val
    hamilteigenvec=eigen[1] #this eigen vector
    hamiltvec1= hamilteigenvec[0] # we wanna get one of each eigenvectors ofc
    hamiltvec2= hamilteigenvec[1]
    #Check if normalized
    n1= np.dot(hamiltvec1,hamiltvec1)
    n2= np.dot(hamiltvec1,hamiltvec2)

#     print ("The n1 is", n1) #uncomment this to check and u can comment it after
#     print ("The n2 is", n2)
    
    
    #H=> U*H_D*U^+
    
    #Hamilitionian matrix diagonalized
    diahamilt= np.diag(hamilteigenval)
    
    # U * U_dagger should equal to 1
    #U_ dagger is the unitary matrix transposed which is just the eigenvectors vertically
    unitary_dagger= np.array([hamiltvec1, hamiltvec2])

    # U which is just the eigenvects horizontal
    unitary_matrix= np.conjugate(np.transpose(unitary_dagger))
    

    #H
    #H_H= np.matmul(Ud_hamilt, np.matmul(diahamilt,U_H))

    #making e^-iLeigenval
#define complex
    iLeval1= complex(0,-L*hamilteigenval[0])
    iLeval2= complex(0,-L*hamilteigenval[1])


#define exp
    e1= np.exp(iLeval1)
    e2= np.exp(iLeval2)

#make matrix of exp
    e_matrix= np.array ([[e1,0],[0,e2]])
    
    
#Hermition matrix

    hermition_matrix= np.matmul(unitary_dagger, np.matmul(e_matrix,unitary_matrix))

    #Olivia = expm(unitary_matrix*baseline)
    
    return hermition_matrix
#once we calc the time evolution, the abs value squared is the probability we care about.


# In[ ]:


Evolution_matrix(2000,0.5,.58,10**-3)


#  ### Probability defined

# In[ ]:


def Prob_2flav (f_i, f_f,L,E,theta,deltamsq):
    
    #Hamiltonian matrix
    hamilt= Hamiltionian_matrix(E,theta,deltamsq)    
    #evolution
    ematrix=Evolution_matrix(L,E,theta,deltamsq)
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix, initial_f))

   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav (1,0,2000,0.5,.58,10**-3))
print("Peu",Prob_2flav (0,1,2000,0.5,.58,10**-3))
print("Pee",Prob_2flav (0,0,2000,0.5,.58,10**-3))
print("Puu",Prob_2flav (1,1,2000,0.5,.58,10**-3))


# ## Plotting

# In[ ]:


energy_a=np.linspace(.5,5,1000)


# In[ ]:


#The arrays in vacuum
P_eu_v= np.array([Prob_2flav (1, 0,2000,energy,.58,10**-3) for energy in (energy_a)])
P_ue_v= np.array([Prob_2flav(0,1,2000,energy,.58,10**-3) for energy in (energy_a)])
P_ee_v= np.array([Prob_2flav(0,0,2000,energy,.58,10**-3) for energy in (energy_a)])
P_uu_v= np.array([Prob_2flav(1,1,2000,energy,.58,10**-3) for energy in (energy_a)])


# In[ ]:


len(P_ee_v)


# ### Plot P_ue and P_eu

# In[ ]:


plt.plot(energy_a,P_eu_v, label='P_ue',color= 'blue',linestyle='--')
plt.plot(energy_a,P_ue_v, label='P_eu',color= 'pink')
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Vacuum')
plt.legend()
plt.show()


# ### Plot P_ee and P_uu
# 

# In[ ]:


plt.plot(energy_a,P_ee_v, label='P_ee',color= 'blue',linestyle='--')
plt.plot(energy_a,P_uu_v, label='P_uu',color= 'pink')
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Vacuum')
plt.legend()
plt.show()


# ## Plot P_ue, P_eu, P_uu, and P_ee

# In[ ]:


plt.plot(energy_a,P_eu_v, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_ue_v, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_v, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_v, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Vacuum vs. Energy')
plt.legend()
plt.savefig("Probability_in_Vacuum")
plt.show()


# # Probability in Constant Matter

# ### Evolution matrix with addition of Matter Potential (A)

# In[ ]:


#e^-i*L*eval

#define L_hamilitonian 
def Evolution_matrix_CM(L,E,theta,deltamsq):
    
    #Matter Potential
    A_matrix= np.array([[1,0],[0,0]])
    # Can mess around with 10^-5 
    hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ A_matrix*(10**-4)*5.067730718
    
    #the eigenvalues(vals) and vectors (vec)
    eigen=LA.eig(hamilt) #so this function shows both vec and vals
    hamilteigenval= eigen[0] #this seperate to get the eigen val
    hamilteigenvec=eigen[1] #this eigen vector
    hamiltvec1= hamilteigenvec[0] # we wanna get one of each eigenvectors ofc
    hamiltvec2= hamilteigenvec[1]
    #Check if normalized
    n1= np.dot(hamiltvec1,hamiltvec1)
    n2= np.dot(hamiltvec1,hamiltvec2)

    #print ("The n1 is", n1) #uncomment this to check and u can comment it after
    #print ("The n2 is", n2)
    
    
    #H=> U*H_D*U^+
    
    #Hamilitionian matrix diagonalized
    diahamilt= np.diag(hamilteigenval)
    
    # U * U_dagger should equal to 1
    #U_ dagger is the unitary matrix transposed which is just the eigenvectors vertically
    unitary_dagger= np.array([hamiltvec1, hamiltvec2])

    # U which is just the eigenvects horizontal
    unitary_matrix= np.conjugate(np.transpose(unitary_dagger))
    

    #H
    #H_H= np.matmul(Ud_hamilt, np.matmul(diahamilt,U_H))

    #making e^-iLeigenval
#define complex
    iLeval1= complex(0,-L*hamilteigenval[0])
    iLeval2= complex(0,-L*hamilteigenval[1])


#define exp
    e1= np.exp(iLeval1)
    e2= np.exp(iLeval2)

#make matrix of exp
    e_matrix= np.array ([[e1,0],[0,e2]])
    
    
#Hermition matrix

    hermition_matrix= np.matmul(unitary_dagger, np.matmul(e_matrix,unitary_matrix))

    
    return hermition_matrix
#once we calc the time evolution, the abs value squared is the probability we care about.


#  ### Probability defined  in Constant Matter Potential

# In[ ]:


def Prob_2flav_CM (f_i, f_f,L,E,theta,deltamsq):
    
    #Matter Potential
    A_matrix= np.array([[1,0],[0,0]])
    # Can mess around with 10^-5 
    hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ A_matrix*(10**-4)*5.067730718 
    #evolution
    ematrix_CM=Evolution_matrix_CM(L,E,theta,deltamsq)
    
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_CM, initial_f))

   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_CM(1,0,2000,0.5,.58,10**-3))
print("Peu",Prob_2flav_CM (0,1,2000,0.5,.58,10**-3))
print("Pee",Prob_2flav_CM (0,0,2000,0.5,.58,10**-3))
print("Puu",Prob_2flav_CM (1,1,2000,0.5,.58,10**-3))


# ## Plotting 

# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_CM= np.array([Prob_2flav_CM (1, 0,2000,energy,.58,10**-3) for energy in (energy_a)])
P_ue_CM= np.array([Prob_2flav_CM (0,1,2000,energy,.58,10**-3) for energy in (energy_a)])
P_ee_CM= np.array([Prob_2flav_CM (0,0,2000,energy,.58,10**-3) for energy in (energy_a)])
P_uu_CM= np.array([Prob_2flav_CM (1,1,2000,energy,.58,10**-3) for energy in (energy_a)])


# ### Plot P_ue and P_eu

# In[ ]:


plt.plot(energy_a,P_eu_CM, label='P_ue',color= 'blue', linestyle='--')
plt.plot(energy_a,P_ue_CM, label='P_eu',color= 'pink')
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Constant Matter Potential vs. Energy')
plt.legend()
plt.show()


# ### Plot P_ee and P_uu

# In[ ]:


plt.plot(energy_a,P_ee_CM, label='P_ee',color= 'blue',linestyle='--')
plt.plot(energy_a,P_uu_CM, label='P_uu',color= 'pink')
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Constant Matter Potential vs. Energy')
plt.legend()
plt.show()


# ## Plot P_ue, P_eu, P_uu, and P_ee

# In[ ]:


plt.plot(energy_a,P_ue_CM, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_CM, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_CM, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_CM, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in Constant Matter Potential vs. Energy')
plt.legend()
plt.savefig("Probability_in_CM")
plt.show()


# In[ ]:


plt.title('Probability in Constant Matter Potential and Vacuum vs. Energy')
plt.plot(energy_a,P_ue_CM, label='P_ue constant matter',color= 'blue',linestyle='--')
plt.plot(energy_a,P_eu_v, label='P_ue vacuum ',color= 'red',linestyle='--')
plt.legend()
plt.savefig("Probability_in_CM_and_Vac")
plt.show()


# # Probability in T-dep Matter (x3)

# In[ ]:


#e^-i*L*eval

#define L_hamilitonian 
def Evolution_matrix_TDM(L,E,theta,deltamsq,scaling_factor):
    
    #Matter Potential
    A_matrix= np.array([[1,0],[0,0]])
    # Can mess around with 10^-5 NOTE:NO multiplication by 3
    hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ (A_matrix*scaling_factor*(10**-4)*5.067730718)
    
    #the eigenvalues(vals) and vectors (vec)
    eigen=LA.eig(hamilt) #so this function shows both vec and vals
    hamilteigenval= eigen[0] #this seperate to get the eigen val
    hamilteigenvec=eigen[1] #this eigen vector
    hamiltvec1= hamilteigenvec[0] # we wanna get one of each eigenvectors ofc
    hamiltvec2= hamilteigenvec[1]
    #Check if normalized
    n1= np.dot(hamiltvec1,hamiltvec1)
    n2= np.dot(hamiltvec1,hamiltvec2)

    #print ("The n1 is", n1) #uncomment this to check and u can comment it after
    #print ("The n2 is", n2)
    
    
    #H=> U*H_D*U^+
    
    #Hamilitionian matrix diagonalized
    diahamilt= np.diag(hamilteigenval)
    
    # U * U_dagger should equal to 1
    #U_ dagger is the unitary matrix transposed which is just the eigenvectors vertically
    unitary_dagger= np.array([hamiltvec1, hamiltvec2])

    # U which is just the eigenvects horizontal
    unitary_matrix= np.conjugate(np.transpose(unitary_dagger))
    

    #H
    #H_H= np.matmul(Ud_hamilt, np.matmul(diahamilt,U_H))

    #making e^-iLeigenval
#define complex
    iLeval1= complex(0,-L*hamilteigenval[0])
    iLeval2= complex(0,-L*hamilteigenval[1])


#define exp
    e1= np.exp(iLeval1)
    e2= np.exp(iLeval2)

#make matrix of exp
    e_matrix= np.array ([[e1,0],[0,e2]])
    
    
#Hermition matrix

    hermition_matrix= np.matmul(unitary_dagger, np.matmul(e_matrix,unitary_matrix))

     
    return hermition_matrix
    
#once we calc the time evolution, the abs value squared is the probability we care about.


# In[ ]:


def Prob_2flav_TDM3 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(400, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(700, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(900, E, theta, deltamsq,6) #6A
    
    #U_increasing= U3U2U1
    # ematrix_product= TDM3*TDM2*TDM1  1*2*3 np.matmul(1,np.matmul(2,3))
   # 321 np.matmul(3,np.matmul(2,1))
    ematrix_TDM_tot= np.matmul(ematrix_TDM3,np.matmul(ematrix_TDM2, ematrix_TDM1))
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDM3(1,0,0.5,.58,10**-3))
print("Peu",Prob_2flav_TDM3 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDM3(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDM3 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDM3= np.array([Prob_2flav_TDM3(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDM3= np.array([Prob_2flav_TDM3 (0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDM3= np.array([Prob_2flav_TDM3(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDM3= np.array([Prob_2flav_TDM3 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDM3, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM3, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM3, label='P_ee',color= 'yellow',linewidth= 2)
plt.plot(energy_a,P_uu_TDM3, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x3 vs. Energy')
plt.legend()
plt.savefig("tdep3.png")
plt.show()


# # Probability in T-dep Matter (x3) Reverse

# In[ ]:


def Prob_2flav_TDMR3 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(400, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(700, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(900, E, theta, deltamsq,6) #6A
    
    #U_decrease/reverse= U1U2U3
    # ematrix_product= TDM1*TDM2*TDM3  1*2*3 np.matmul(1,np.matmul(2,3))
    ematrix_TDM_tot= np.matmul(ematrix_TDM1,np.matmul(ematrix_TDM2, ematrix_TDM3))
    

    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())

# L = 1
# scaling_factor = 1


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDMR3(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDMR3 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDMR3(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDMR3 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDMR3= np.array([Prob_2flav_TDMR3(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDMR3= np.array([Prob_2flav_TDMR3 (0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDMR3= np.array([Prob_2flav_TDMR3(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDMR3= np.array([Prob_2flav_TDMR3 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDMR3, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR3, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR3, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR3, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x3 vs. Energy Reverse')
plt.legend()
plt.savefig("tdep3R.png")
plt.show()


# In[ ]:


plt.subplot(211) 

plt.plot(energy_a,P_ue_TDM3, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM3, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM3, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM3, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title(' T-dep Matter x3 vs. Energy')
#plt.legend()
plt.subplot(212) 

plt.plot(energy_a,P_ue_TDMR3, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR3, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR3, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR3, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title(' Reverse')
#plt.legend()
plt.savefig("tdep3compare.png")
plt.show()


# # Probability in T-dep Matter (x2)

# In[ ]:


def Prob_2flav_TDM2 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(800, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(1200, E, theta, deltamsq,3) #3A
    
    #U_increasing= U2U1
    # ematrix_product= TDM2*TDM1  1*2*3 np.matmul(1,np.matmul(2,3))
   # 21 np.matmul(2,1)
    ematrix_TDM_tot= np.matmul(ematrix_TDM2, ematrix_TDM1)
    

    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())

# L = 1
# scaling_factor = 1


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDM2(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDM2 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDM2(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDM2 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDM2= np.array([Prob_2flav_TDM2(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDM2= np.array([Prob_2flav_TDM2 (0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDM2= np.array([Prob_2flav_TDM2(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDM2= np.array([Prob_2flav_TDM2 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDM2, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM2, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM2, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM2, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x2 vs. Energy')
plt.legend()
plt.savefig("tdep2.png")
plt.show()


# # Probability in T-dep Matter (x2) Reverse

# In[ ]:


def Prob_2flav_TDMR2 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(800, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(1200, E, theta, deltamsq,3) #3A
    
    #U_decrease/reverse= U1U2
    # ematrix_product= TDM1*TDM2*TDM3  1*2*3 np.matmul(1,np.matmul(2,3))
    ematrix_TDM_tot= np.matmul(ematrix_TDM1, ematrix_TDM2)
    

    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDMR2(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDMR2 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDMR2(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDMR2 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDMR2= np.array([Prob_2flav_TDMR2(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDMR2= np.array([Prob_2flav_TDMR2 (0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDMR2= np.array([Prob_2flav_TDMR2(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDMR2= np.array([Prob_2flav_TDMR2 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDMR2, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR2, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR2, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR2, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x2 vs. Energy Reverse')
plt.legend()
plt.savefig("tdep2R.png")
plt.show()


# In[ ]:


plt.subplot(211) 
plt.plot(energy_a,P_ue_TDM2, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM2, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM2, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM2, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.ylabel('Probability')
plt.title(' T-dep Matter x2 vs. Energy')
#plt.legend()
plt.subplot(212) 
plt.plot(energy_a,P_ue_TDMR2, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR2, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR2, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR2, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title(' Reverse')
#plt.legend()
plt.savefig("tdep2compare.png")
plt.show()


# # Probability in T-dep Matter x4

# In[ ]:


def Prob_2flav_TDM4 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(300, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(400, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(500, E, theta, deltamsq,6) #6A
    ematrix_TDM4= Evolution_matrix_TDM(800, E, theta, deltamsq,9) #9A

    
    #U_increasing= U4U3U2U1
    # ematrix_product= TDM3*TDM2*TDM1  1*2*3 np.matmul(1,np.matmul(2,3))
    # 321 np.matmul(3,np.matmul(2,1))
    ematrix_TDM_tot = np.matmul(ematrix_TDM4, np.matmul(ematrix_TDM3, np.matmul(ematrix_TDM2, ematrix_TDM1)))
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDM4(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDM4 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDM4(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDM4 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDM4= np.array([Prob_2flav_TDM4(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDM4= np.array([Prob_2flav_TDM4(0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDM4= np.array([Prob_2flav_TDM4(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDM4= np.array([Prob_2flav_TDM4 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDM4, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM4, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM4, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM4, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x4 vs. Energy')
plt.legend()
plt.savefig("tdep4.png")
plt.show()


# # Probability in T-dep Matter Reverse x4

# In[ ]:


def Prob_2flav_TDMR4 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
   
    ematrix_TDM1= Evolution_matrix_TDM(300, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(400, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(500, E, theta, deltamsq,6) #6A
    ematrix_TDM4= Evolution_matrix_TDM(800, E, theta, deltamsq,9) #9A

    
    #U_decrease/reverse= U1U2U3
    # ematrix_product= TDM1*TDM2*TDM3  1*2*3 np.matmul(1,np.matmul(2,3))
    ematrix_TDM_tot = np.matmul(ematrix_TDM1, np.matmul(ematrix_TDM2, np.matmul(ematrix_TDM3, ematrix_TDM4)))

    

    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())

# L = 1
# scaling_factor = 1


# In[ ]:


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDMR4(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDMR4 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDMR4(0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDMR4 (1,1,0.5,.58,10**-3))


# In[ ]:


#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDMR4= np.array([Prob_2flav_TDMR4(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDMR4= np.array([Prob_2flav_TDMR4 (0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDMR4= np.array([Prob_2flav_TDMR4(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDMR4= np.array([Prob_2flav_TDMR4 (1,1,energy,.58,10**-3)for energy in (energy_a)])


# In[ ]:


plt.plot(energy_a,P_ue_TDMR4, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR4, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR4, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR4, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x4 vs. Energy')
plt.legend()
plt.savefig("tdep4R.png")
plt.show()


# In[ ]:


plt.subplot(211)
plt.plot(energy_a,P_ue_TDM4, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM4, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM4, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM4, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.ylabel('Probability')
plt.title(' T-dep Matter x4 vs. Energy')
#plt.legend()
plt.subplot(212) 
plt.plot(energy_a,P_ue_TDMR4, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR4, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR4, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR4, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title(' Reverse')
#plt.legend()
plt.savefig("tdep4compare.png")
plt.show()


# In[ ]:


def Prob_2flav_TDM5 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(100, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(200, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(400, E, theta, deltamsq,6) #6A
    ematrix_TDM4= Evolution_matrix_TDM(600, E, theta, deltamsq,9) #9A
    ematrix_TDM5= Evolution_matrix_TDM(700, E, theta, deltamsq,12) #12A


    
    #U_increasing= U4U3U2U1
    # ematrix_product= TDM3*TDM2*TDM1  1*2*3 np.matmul(1,np.matmul(2,3))
    # 321 np.matmul(3,np.matmul(2,1))
    ematrix_TDM_tot = np.matmul(ematrix_TDM5, np.matmul(ematrix_TDM4, np.matmul(ematrix_TDM3, np.matmul(ematrix_TDM2, ematrix_TDM1))))
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDM5(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDM5 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDM5 (0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDM5 (1,1,0.5,.58,10**-3))

#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDM5= np.array([Prob_2flav_TDM5(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDM5= np.array([Prob_2flav_TDM5(0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDM5= np.array([Prob_2flav_TDM5(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDM5= np.array([Prob_2flav_TDM5 (1,1,energy,.58,10**-3)for energy in (energy_a)])


plt.plot(energy_a,P_ue_TDM5, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM5, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM5, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM5, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter x5 vs. Energy')
plt.legend()
plt.savefig("tdep5.png")
plt.show()


# In[ ]:


def Prob_2flav_TDMR5 (f_i, f_f,E,theta,deltamsq):
    
#     #Matter Potential
#     A_matrix= np.array([[1,0],[0,0]])
#     # Can mess around with 10^-5 
#     hamilt= Hamiltionian_matrix(E,theta,deltamsq)+ ( A_matrix*scaling_factor*(10**-4)*5.067730718 )
#     #evolution
#     ematrix_TDM = Evolution_matrix_TDM(L, E, theta, deltamsq,scaling_factor)
   
    #Different evolution matrices
    
    ematrix_TDM1= Evolution_matrix_TDM(100, E, theta, deltamsq,1) #1A
    ematrix_TDM2= Evolution_matrix_TDM(200, E, theta, deltamsq,3) #3A
    ematrix_TDM3= Evolution_matrix_TDM(400, E, theta, deltamsq,6) #6A
    ematrix_TDM4= Evolution_matrix_TDM(600, E, theta, deltamsq,9) #9A
    ematrix_TDM5= Evolution_matrix_TDM(700, E, theta, deltamsq,12) #12A


    
    #U_increasing= U4U3U2U1
    # ematrix_product= TDM3*TDM2*TDM1  1*2*3 np.matmul(1,np.matmul(2,3))
    # 321 np.matmul(3,np.matmul(2,1))
    ematrix_TDM_tot = np.matmul(ematrix_TDM1, np.matmul(ematrix_TDM2, np.matmul(ematrix_TDM3, np.matmul(ematrix_TDM4, ematrix_TDM5))))
    
    #Unit vectors for the flavors
    FV_1= np.array ([1,0])#elec
    FV_2= np.array ([0,1])#muon
    FV= np.array ([FV_1,FV_2])
    
    #flavors
    initial_f= np.transpose(FV[f_i])
    final_f= FV[f_f]
    
    # multiply final flavor *U*initial flavor* time evolution matrix
    h1= np.matmul(final_f,np.matmul(ematrix_TDM_tot, initial_f))
   
   #complex conjugate (previous result)
    h2= np.conj(h1)

    #Probability= result 2 * result 1 or a*a'
    
    probability= h1*h2
    
    #print("this is check", h1)
    
    return np.real(probability.item())


#check P_ue=P_eu, P_ee=P_uu
print("Pue",Prob_2flav_TDMR5(1,0,0.5,.58,10**-3,))
print("Peu",Prob_2flav_TDMR5 (0,1,0.5,.58,10**-3))
print("Pee",Prob_2flav_TDMR5 (0,0,0.5,.58,10**-3))
print("Puu",Prob_2flav_TDMR5 (1,1,0.5,.58,10**-3))

#The arrays NOTE: energy_a=np.linspace(.5,5) from vacuum
P_eu_TDMR5= np.array([Prob_2flav_TDMR5(1,0,energy,.58,10**-3) for energy in (energy_a)])
P_ue_TDMR5= np.array([Prob_2flav_TDMR5(0,1,energy,.58,10**-3) for energy in (energy_a)])
P_ee_TDMR5= np.array([Prob_2flav_TDMR5(0,0,energy,.58,10**-3) for energy in (energy_a)])
P_uu_TDMR5= np.array([Prob_2flav_TDMR5 (1,1,energy,.58,10**-3)for energy in (energy_a)])


plt.plot(energy_a,P_ue_TDMR5, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR5, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR5, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR5, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title('Probability in T-dep Matter Reverse x5 vs. Energy')
plt.legend()
plt.savefig("tdep5R.png")
plt.show()


# In[ ]:


plt.subplot(211)
plt.plot(energy_a,P_ue_TDM5, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDM5, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDM5, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDM5, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.ylabel('Probability')
plt.title(' T-dep Matter x5 vs. Energy')
#plt.legend()
plt.subplot(212) 
plt.plot(energy_a,P_ue_TDMR5, label='P_ue',color= 'blue',linestyle='--', linewidth= 3)
plt.plot(energy_a,P_eu_TDMR5, label='P_eu',color= 'pink', linewidth= 2)
plt.plot(energy_a,P_ee_TDMR5, label='P_ee',color= 'yellow',  linewidth= 2)
plt.plot(energy_a,P_uu_TDMR5, label='P_uu',color= 'red',linestyle='--',  linewidth= 3)
plt.xlabel('Energy (GeV)')
plt.ylabel('Probability')
plt.title(' Reverse')
#plt.legend()
plt.savefig("tdep5compare.png")
plt.show()

