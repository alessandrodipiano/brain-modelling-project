import numpy as np
from brian2 import *


# Unit conversions (paper uses per cm^2; Brian2 wants SI per m^2)

# 1 cm^2 = 1e-4 m^2
# so X / cm^2  ==  X * 1e4 / m^2
#
# 1 uA/cm^2 = 1e-6 A * 1e4 / m^2 = 1e-2 A/m^2
# 1 mS/cm^2 = 1e-3 S * 1e4 / m^2 = 10 S/m^2
# 1 uF/cm^2 = 1e-6 F * 1e4 / m^2 = 1e-2 F/m^2
UA_PER_CM2_TO_SI = 1e-2 * amp / meter**2
MS_PER_CM2_TO_SI = 10.0 * siemens / meter**2
UF_PER_CM2_TO_SI = 1e-2 * farad / meter**2


# Network size + connectivity
N_E, N_I = 200, 50
P_EE, P_EI, P_IE, P_II = 0.10, 0.40, 0.50, 0.60

# Capacitance
C_E = 1.0 * UF_PER_CM2_TO_SI
C_I = 1.0 * UF_PER_CM2_TO_SI

# Leak
EL  = -65.0 * mV
gL_E = 0.05 * MS_PER_CM2_TO_SI
gL_I = 0.50 * MS_PER_CM2_TO_SI

# Thresholds / reset
VT_E = -45.0 * mV
VT_I = -30.0 * mV
VR   = -52.0 * mV
V_peak = 20 * mV


# Adaptation (E only)
a = 0.01 / ms          # paper form uses fixed decay
d = 0.2 * UA_PER_CM2_TO_SI

# Synaptic time constants
tau_nmda = 80.0 * ms
tau_ampa_EE = 3.0 * ms
tau_ampa_EI = 1.0 * ms
tau_gaba = 2.0 * ms

# Reversal potentials
E_AMPA = 0.0 * mV
E_NMDA = 0.0 * mV
E_GABA = -70.0 * mV

# Synaptic conductances (Table 2)

gNE = 0.008 * MS_PER_CM2_TO_SI
gNI = 0.008 * MS_PER_CM2_TO_SI
gEE = 0.10  * MS_PER_CM2_TO_SI
gEI = 0.08  * MS_PER_CM2_TO_SI
gIE = 0.25  * MS_PER_CM2_TO_SI
gII = 0.10  * MS_PER_CM2_TO_SI

# Magnesium concentration
Mg = 1.0

# External tonic drive
I_app_E = 4.0 * UA_PER_CM2_TO_SI
I_app_I = 0.0 * UA_PER_CM2_TO_SI

eqs_E = """
# Intrinsic quadratic term (as in the paper's QIF-like formulation)
I_intrinsic = gL * (v - EL) * (v - VT) / (VT - EL) : amp/meter**2
I_adapt = z : amp/meter**2

# NMDA magnesium block
B = 1.0 / (1.0 + (Mg * exp(-0.062 * (v/mV)) / 3.57)) : 1

# Synaptic currents: E receives from E (AMPA+NMDA) and from I (GABA)
I_ampa = gEE * sA_EE * (v - E_AMPA) : amp/meter**2
I_nmda = gNE * sN_EE * B * (v - E_NMDA) : amp/meter**2
I_gaba = gIE * sG_IE * (v - E_GABA) : amp/meter**2
I_syn = I_ampa + I_nmda + I_gaba : amp/meter**2

# Membrane
dv/dt = (I_app + I_intrinsic - I_adapt - I_syn) / C : volt

# Adaptation
dz/dt = -a * z : amp/meter**2

# Synaptic gate decays
dsA_EE/dt = -sA_EE / tau_ampa_EE : 1
dsN_EE/dt = -sN_EE / tau_nmda : 1
dsG_IE/dt = -sG_IE / tau_gaba : 1

# Per-neuron parameters / inputs
I_app : amp/meter**2
C : farad/meter**2
gL : siemens/meter**2
VT : volt
"""

eqs_I = """
I_intrinsic = gL * (v - EL) * (v - VT) / (VT - EL) : amp/meter**2
I_adapt = 0*amp/meter**2 : amp/meter**2  # no adaptation current in I

B = 1.0 / (1.0 + (Mg * exp(-0.062 * (v/mV)) / 3.57)) : 1

# Synaptic currents: I receives from E (AMPA+NMDA) and from I (GABA)
I_ampa = gEI * sA_EI * (v - E_AMPA) : amp/meter**2
I_nmda = gNI * sN_EI * B * (v - E_NMDA) : amp/meter**2
I_gaba = gII * sG_II * (v - E_GABA) : amp/meter**2
I_syn = I_ampa + I_nmda + I_gaba : amp/meter**2

dv/dt = (I_app + I_intrinsic - I_adapt - I_syn) / C : volt

# Gate decays
dsA_EI/dt = -sA_EI / tau_ampa_EI : 1
dsN_EI/dt = -sN_EI / tau_nmda : 1
dsG_II/dt = -sG_II / tau_gaba : 1

I_app : amp/meter**2
C : farad/meter**2
gL : siemens/meter**2
VT : volt
"""




def network(alpha_EI, dt, rng_seed=0,gNI_mS_cm2=0.008):
    global gNI  # because eqs_I refers to the Python symbol gNI
    start_scope()
    np.random.seed(rng_seed)
    seed(rng_seed) 
    
    gNI = gNI_mS_cm2 * MS_PER_CM2_TO_SI

    defaultclock.dt = dt * ms
    method = "euler"
    

    E = NeuronGroup(
        N_E, eqs_E,
        threshold="v >= V_peak",
        reset="v = VR; z += d",
        method=method,
        name="E"
    )

    I = NeuronGroup(
        N_I, eqs_I,
        threshold="v >= V_peak",
        reset="v = VR",          # no adaptation jump in I
        method=method,
        name="I"
    )

    # Set per-population intrinsic parameters
    E.C = C_E
    E.gL = gL_E
    E.VT = VT_E
    E.I_app = I_app_E

    I.C = C_I
    I.gL = gL_I
    I.VT = VT_I
    I.I_app = I_app_I

    # Initial conditions
    E.v = EL + 1*mV*randn(N_E)
    E.z = 0.0 * amp/meter**2
    E.sA_EE = 0.0
    E.sN_EE = 0.0
    E.sG_IE = 0.0

    I.v = EL + 1*mV*randn(N_I)
    I.sA_EI = 0.0
    I.sN_EI = 0.0
    I.sG_II = 0.0

    # Weights (dimensionless gate jumps)
    K_EE = P_EE * (N_E - 1)
    K_EI = P_EI * N_E
    K_IE = P_IE * N_I
    K_II = P_II * (N_I - 1)

    wA_EE = 1.0 / K_EE
    wN_EE = 1.0 / K_EE
    wA_EI = 1.0 / K_EI
    wN_EI = 1.0 / K_EI
    wG_IE = 1.0 / K_IE
    wG_II = 1.0 / K_II

    # E -> E (paired AMPA+NMDA on same edges)
    S_EE = Synapses(E, E, model="wA:1\nwN:1", on_pre="sA_EE_post += wA; sN_EE_post += wN")
    S_EE.connect(condition='i != j', p=P_EE)
    S_EE.wA = wA_EE
    S_EE.wN = wN_EE

    # E -> I
    S_EI = Synapses(E, I, model="wA:1\nwN:1", on_pre="sA_EI_post += wA; sN_EI_post += wN")
    S_EI.connect(p=P_EI)


    S_EI.wA = alpha_EI
    S_EI.wN = alpha_EI

    # I -> E
    S_IE = Synapses(I, E, model="wG:1", on_pre="sG_IE_post += wG")
    S_IE.connect(p=P_IE)
    S_IE.wG = wG_IE

    # I -> I
    S_II = Synapses(I, I, model="wG:1", on_pre="sG_II_post += wG")
    S_II.connect(condition='i != j', p=P_II)
    S_II.wG = wG_II

    for S in (S_EE, S_EI, S_IE, S_II):
        S.delay = 0.5 * ms


    # ----------------------------
    # Recording
    # ----------------------------
    spE = SpikeMonitor(E)
    spI = SpikeMonitor(I)

    vE = StateMonitor(E, "v", record=range(5))
    vI = StateMonitor(I, "v", record=range(5))

    T = 2000*ms
    run(T)

    t0 = 200*ms
    rate_E_ss = np.sum(spE.t >= t0) / ((T - t0)/second) / N_E
    rate_I_ss = np.sum(spI.t >= t0) / ((T - t0)/second) / N_I
    vmax0_mV = float(np.max(vE.v[0] / mV))

    return dict(alpha_EI=alpha_EI, dt=float(dt/ms),
                rate_E_ss=float(rate_E_ss), rate_I_ss=float(rate_I_ss),
                I_app_E=float(E.I_app[0] / (amp/meter**2)),
                I_app_I=float(I.I_app[0] / (amp/meter**2)),
                vmax0_mV=vmax0_mV)
    