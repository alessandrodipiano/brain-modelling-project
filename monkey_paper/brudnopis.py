#   FORM OF THE DATA ARRAY 
# | meanL - meanR | sdL = sdR | 1 if choice==LEFT 0 otherwise | 1 if choice is correct 0 otherwise | 1 if lowSD, 0 otherwise |


from brian2 import *
prefs.codegen.target = 'cython'

import importlib
import numpy as np
import concurrent.futures

import sim2
import utils
importlib.reload(sim2)
importlib.reload(utils)

from sim2 import network
from utils import create_input, mean_rate
from utils_pittore import trial_features 

def find_r(SM, dt=0.001):
    """
    Computes time dependent firing rate using the causal exponential filter;
    INPUT: SM brian2 spike monitor; dt time bin length
    OUTPUT: array of time points, array of associated spike rates
    """
    N = 240 
    sim_t = 5.0 
    tau = 0.02  

    spike_times = np.asarray(SM.t / second)

    bins = np.arange(0, sim_t + dt, dt)
    counts, bin_edges = np.histogram(spike_times, bins=bins)
    counts = counts / N
    
    t_kernel = np.arange(0, 5 * tau, dt)
    kernel = (1.0/tau) * np.exp(-t_kernel / tau)
    
    filtered_rate = np.convolve(counts, kernel, mode='full')[:len(counts)]
    time_vector = bin_edges[:-1]
    
    return time_vector, filtered_rate

def run_single_trial(trial):

    ev_st = trial[0].mean() - trial[1].mean()
    st_diff = trial[0].std() - trial[1].std()

    mu1, mu2 = mean_rate(trial)
    pop1, pop2, R1, R2, E1, E2 = network(mu1, mu2) 

    time1, rate1 = find_r(pop1)
    time2, rate2 = find_r(pop2)

    final1 = rate1[-1]
    final2 = rate2[-1]

    sd_val = int(trial[2] == trial[3]) 
    choice = int(final1 > final2) 
    correct = int(choice != trial[3]) 

    return (ev_st, st_diff, choice, correct, sd_val)

if __name__ == '__main__':
    
    n_trials = 200
    trials = create_input(n_trials)
    
    results = []
    
    print(f"Starting simulation of {n_trials} trials across multiple cores...")
    
    # ProcessPoolExecutor automatically uses all available logical CPU cores.
    # You can specify max_workers=4 if you want to limit the core usage.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map applies the function to each element in the trials list
        for i, result in enumerate(executor.map(run_single_trial, trials), 1):
            results.append(result)
            if i % 10 == 0:
                print(f"Completed {i}/{n_trials} trials")

    # Stack results and save
    results_array = np.array(results)
    np.save('trials_control2.npy', results_array) 

    data = np.load('trials_control2.npy')
    print(data)