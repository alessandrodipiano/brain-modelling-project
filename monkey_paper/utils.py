#NARROW-BROAD TRIAL 
#in each trial there is a high sd and a low sd option 

#a pair of 8 realized values from a gaussian distribution forms one trial (two simulations)
#each realized value is an h from range [0, 100]

#The trial generation process was constrained so the samples reasonably reflected the generative parameters. These restrictions required bar heights to range from 1 to 99, and the actual σ for each stream to be no more than 4 from the generative value

#TYPE ONE 
#Z uniform (-0.25, 0.25)
#mu = 50 + Z*sigma
#low var: sigma=12; high_var: sigma=24

#TYPE TWO
#the means are fixed to control the levels of correctness
#narrow correct: muN unif(48, 60) sigmaN=12  and muB = muN - 8, sigmaB=24
#broad correct vice versa

#ambiguous mu unif(44, 56) muB = muN

# population 1 always receives broad and 2 always receives narrow

import numpy as np
import matplotlib.pyplot as plt

def create_input(n):
    '''generates n input samples. each input sample is 2 arrays of 8 bar heights each, one generated with high std (broad, 24), the other low std (narrow, 12). samples are gaussian with different means; 
    constraints on generated data: falls in the [1, 99] range; sample std is close to the std
    used at generation
    OUTPUT: [array1, array2, I(array2 is the narrow sample), I(narrow sample correct), X= strength of evidence]
    '''

    trials = []
    while len(trials) < n:

        #generate means for the narrow and broad samples (effectively determines which will be 
        # the correct choise (in the choose high task))
        Z = np.random.uniform(-0.25, 0.25)
        muN = 50 + Z*12
        muB = 50 + Z*24

        #for each mean generate 8 bars (gaussian) 
        hN = np.random.normal(muN, 12, 8)
        hB = np.random.normal(muB, 24, 8)

        #check if realized sample reasonably reflects parameters
        cond1 = np.all(hN) >= 1 and np.all(hN) <= 99 and np.all(hB) >= 1 and np.all(hB) <= 99
        cond2 = np.abs(np.std(hN) - 12) < 4 and np.abs(np.std(hB) - 24) < 4

        if cond1 and cond2: 
            trial_data = [0, 0, 0, 0, 0] 

            posN = np.random.randint(0, 2) #determine the position of the narrow sample 
            posB = np.abs(posN - 1)
            trial_data[posN] = hN 
            trial_data[posB] = hB  

            trial_data[2] = posN # stores info about which sample is narrow
            trial_data[3] = int(np.mean(hN) > np.mean(hB)) # 1 if narrow correct

            #strength of evidence is defined as difference in means of the correct and incorrect choice 
            trial_data[4] = np.abs(np.mean(hN) - np.mean(hB))

            trials.append(trial_data)
    return trials 

def plot_evidence_histogram(bins=30):
    '''generates trials using create_input() and computes a histogram of evidence strength.
    Evidence strength is defined as the absolute difference in means between the correct and incorrect choice.
    
    Args:
        num_trials: number of trials to generate (note: create_input generates 5 trials at a time)
        bins: number of bins for the histogram
    
    Returns:
        evidence_values: array of evidence strength values from all trials
    '''
    trials = create_input(1000)
    evidence_values = [trial[4] for trial in trials]
    
    plt.figure(figsize=(10, 6))
    plt.hist(evidence_values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Evidence Strength (difference in means)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Evidence Strength Across Trials', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return evidence_values

def mean_rate(trial_data):
    '''create array of mean firing rates (Poisson r.v.) that will serve as input to group 1 and group 2 neurons in the network'''

    mu0 = 30.0 
    mu1 = 1.0

    muPois1 = mu0 + mu1*(trial_data[0] - 50)
    muPois2 = mu0 + mu1*(trial_data[1] - 50)

    muPois1[muPois1 < 0] = 0
    muPois2[muPois2 < 0] = 0

    return muPois1, muPois2

#plot_evidence_histogram()
