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
from scipy.optimize import minimize
import statsmodels.api as sm

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

            trial_data[2] = posN # index of the narrow sample
            trial_data[3] = int(np.mean(trial_data[1]) > np.mean(trial_data[0])) # index of the correct choice

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

def trial_features(trial_data):
    '''
    trial_data[0] = stream 0 bars (len 8)
    trial_data[1] = stream 1 bars (len 8)

    We assume stream 0 = LEFT, stream 1 = RIGHT .
    '''
    left = np.asarray(trial_data[0], dtype=float)
    right = np.asarray(trial_data[1], dtype=float)

    mL, mR = left.mean(), right.mean()
    sL, sR = left.std(ddof=0), right.std(ddof=0)

    return {
        "mean_diff": abs(mL - mR),
        "sd_diff": sL - sR,
        "mL": mL, "mR": mR,
        "sL": sL, "sR": sR
    }

def compute_pvb(X, y):
    """
    X[:,0] = mean_diff (mL - mR)
    X[:,1] = sd_diff   (sL - sR)
    y      = 1 if choose LEFT else 0
    """
    X_design = sm.add_constant(X)  # adds intercept beta0
    model = sm.Logit(y, X_design).fit(disp=False) #fit logistic regression

    beta0, beta_mean, beta_sd = model.params #extract paramms
    pvb = beta_sd / beta_mean

    return {
        "beta0": float(beta0),
        "beta_mean": float(beta_mean),
        "beta_sd": float(beta_sd),
        "pvb": float(pvb),
        "summary": model.summary().as_text()
    }



def psychometric_P(x, alpha, beta):  #P(x)=0.5 + 0.5*(1 - exp(-(x/alpha)^beta))  >-- psychometric function used in paper
    x = np.asarray(x, dtype=float)
    alpha = float(alpha)
    beta = float(beta)
    return 0.5 + 0.5 * (1.0 - np.exp(- (x / alpha) ** beta))


def neg_log_likelihood(params, x, y):  #log lilekihood for bernoulli 
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    p = psychometric_P(x, alpha, beta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_psychometric_mle(x, y): #fitting MLE estimator
    x = np.asarray(x, dtype=float) #SoE
    y = np.asarray(y, dtype=int) #accuracy

    x_pos = x[x > 0]  #initialization w guesses
    alpha0 = float(np.median(x_pos)) if x_pos.size else 1.0
    beta0 = 2.0
    bounds = [(1e-3, max(1.0, float(x.max()) * 10.0)), (0.1, 20.0)] #we put bound to avoid beta jumping

    res = minimize(
        neg_log_likelihood, #we minimize the likelihood to find optimal parameters
        x0=np.array([alpha0, beta0], dtype=float),
        args=(x, y),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000}
    )

    alpha_hat, beta_hat = res.x
    return float(alpha_hat), float(beta_hat)


def binned_accuracy(x, y, n_bins=10):  #used to get avarage accuracy per range of evidence
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)) #each bin gets similar numb of trials
    edges = np.unique(edges)

    # if many repeated edges (e.g., many identical x), fallback to linear bins
    if len(edges) < 3:
        edges = np.linspace(float(x.min()), float(x.max()) + 1e-9, n_bins + 1)

    centers, acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)

        n = int(mask.sum())
        if n == 0:
            continue

        centers.append(0.5 * (lo + hi))
        acc.append(float(y[mask].mean()))
        counts.append(n)

    return np.array(centers), np.array(acc), np.array(counts)



#plot_evidence_histogram()
