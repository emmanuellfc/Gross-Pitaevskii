import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random

def calc_s0(psi_init, h, m, g, mu, k, x_max, x_min, steps,x_vec,temp,potential):
    
    #normalize psi_init
    dx=(x_max-x_min)/steps
    psi_init=psi_init/np.sqrt(np.sum(np.abs(psi_init)**2)*dx)

    beta=1/k*temp

    #laplacian
    laps=[]
    for i in range(1,len(psi_init)-1):
        laplacian=-h**2/(2*m)*(psi_init[i+1]-2*psi_init[i]+psi_init[i-1])*psi_init[i]/((x_max-x_min)/steps)**2
        laps.append(laplacian.copy())
    laps_term=laps*np.conjugate(psi_init[1:-1])
    laps_int=integrate.simpson(laps_term, x=x_vec[1:-1])
    print("Laplacian Term: ", laps_int)

    #Potential term
    potential_term=potential*np.conjugate(psi_init)*psi_init
    potential_integrate=integrate.simpson(potential_term, x=x_vec)
    print("Potential Term: ", potential_integrate)

    #Norm^4
    norm4=(np.conjugate(psi_init)*psi_init)**2
    #print(norm4)
    norm4_term=integrate.simpson(norm4, x=x_vec)
    print("Norm^4 Term: ", norm4_term)

    #Norm^2 (part of number operation)
    norm2=np.conjugate(psi_init)*psi_init #I feel like this needs to explicitly be something times complex conj...
    norm2_term=integrate.simpson(norm2, x=x_vec)
    print("Norm^2 Term: ", norm2_term)
    print(type(norm2_term))

    S_0=-beta*((laps_int+potential_integrate+(g/2)*norm4_term)-mu*norm2_term)

    return S_0

def check_func(s0_final, s0_init, psi_new,psi_old,x_max, x_min, steps):

    dx=(x_max-x_min)/steps
    a=np.exp(s0_final-s0_init)
    print("value of a is: ", a)

    if a>=1:
        pass
        print("Candidate state is more probable than seeding state. Candidate state accepted.")
        accepted_s=s0_final
        psi_out=psi_new
    if a<1:
        rand=random.random() #generate random number between 0,1
        print("rand value is: ",rand)
        if rand<=a:
        #if rand >=.5:
            psi_out=psi_new
            print("From random number generation, candidate state is accepted.")
            accepted_s=s0_final
        else:
            pass
            print("From random number generation, candidate state is rejected.")
            accepted_s=s0_init
            psi_out=psi_old

    #normalize psi_out
    # psi_out_norm=psi_out/np.sqrt(np.sum(np.abs(psi_out)**2)*dx)
    
    return accepted_s, psi_out

def normalize(wave_function, x_start, x_end, num_steps):
    """
    Normalize the wave function.
        Parameters: 
            - Wave Function: array
            - x_start
            - x_end
            - dx
            - num steps
    """
    dx = (x_end-x_start)/num_steps
    psi_normalized = wave_function/np.sqrt(np.sum(np.abs(wave_function)**2)*dx)
    return psi_normalized


def loop_stochastic(h, m, g, k, mu, c1, c2, c3, v, phi, u, x_max, x_min, nsteps, xgrid, initial_wave_func, potential_func, iterations, temp):

    #Define initial and boundary conditions
    iter=0
    entropy_store=[]
    psi_store=[]
    psi_sq_store=[]
    psi_store.append(initial_wave_func.copy())

    while iter<iterations:

        print("Iteration: ", iter)

        #calculate inital entropy
        s0_init=calc_s0(initial_wave_func, h, m, g, mu, k, x_max, x_min, nsteps,xgrid, temp, potential_func)
        print("Initial reduced entropy: ",s0_init)

        # PERTURBATION
        rand=random.choice([0,1])

        if rand==0:
            #print("Generating density perturbation...")
            psi=initial_wave_func*(1+c1*v*np.sin(k*xgrid+phi))
        if rand==1:
            psi=initial_wave_func
        #     #print("Generating phase perturbation...")
        #     psi=initial_wave_func*np.exp((1j*c2*v*np.sin(k*xgrid+phi)))

        # Vary particle number
        psi=(1+c3*u)*psi
        #print("new psi is:",psi)

        # Calculate reduced energy of perturbed field
        s0_final=calc_s0(psi, h, m, g, mu, k, x_max, x_min, nsteps,xgrid, temp, potential_func)
        print("Final reduced entropy: ", s0_final)

        # Accept or reject perturbation
        accepted_s, initial_wave_func=check_func(s0_final, s0_init, psi, initial_wave_func, x_max, x_min, nsteps)
        entropy_store.append(accepted_s)
        psi_store.append(initial_wave_func)

        initial_wave_func = normalize(initial_wave_func, x_min, x_max, nsteps)
        
        psi_sq=np.conjugate(initial_wave_func)*initial_wave_func
        
        psi_sq = normalize(psi_sq, x_min, x_max, nsteps)
        psi_sq_store.append(psi_sq.copy())
        
        iter+=1
    return psi_store, psi_sq_store, entropy_store