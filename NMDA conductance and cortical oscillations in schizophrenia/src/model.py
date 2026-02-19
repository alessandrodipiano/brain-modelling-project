import numpy as np
from brian2 import *

UA_PER_CM2_TO_SI = 1e-2 * amp / meter**2
MS_PER_CM2_TO_SI = 10.0 * siemens / meter**2
UF_PER_CM2_TO_SI = 1e-2 * farad / meter**2

N_E, N_I = 200, 50
P_EE, P_EI, P_IE, P_II = 0.10, 0.40, 0.50, 0.60

EL   = -65.0 * mV
VT_E = -45.0 * mV
VT_I = -30.0 * mV
VR   = -52.0 * mV
V_peak = 20.0 * mV

C_E  = 1.0 * UF_PER_CM2_TO_SI
C_I  = 1.0 * UF_PER_CM2_TO_SI
gL_E = 0.05 * MS_PER_CM2_TO_SI
gL_I = 0.50 * MS_PER_CM2_TO_SI

V_K = -75.0 * mV
a_adapt = 0.01 / ms

# IMPORTANT: z must have units of conductance density (so that z*(v - V_K) is a current density)
# The paper increments z by d=0.2 (Table 1) but does not state units explicitly; consistency with Eq. (2)
# implies z is a conductance-like term. Here we interpret d as 0.2 mS/cm^2.
d_adapt = 0.2 * MS_PER_CM2_TO_SI

tau_n  = 80.0 * ms
tau_e  = 3.0 * ms
tau_ei = 1.0 * ms
tau_i  = 2.0 * ms

Vex = 0.0 * mV
Vin = -70.0 * mV

Mg = 1.0

sigma_E = 1.0 * UA_PER_CM2_TO_SI * sqrt(ms)
sigma_I = 0.8 * UA_PER_CM2_TO_SI * sqrt(ms)


def network(
    dt_ms=0.05,
    T_ms=10000.0,
    rng_seed=0,
    gNI_mS_cm2=0.008,
    gEI_mS_cm2=0.08,
    gNE_mS_cm2=0.008,
    gEE_mS_cm2=0.10,
    gIE_mS_cm2=0.25,
    gII_mS_cm2=0.10,
    Iapp_E_uAcm2=4.0,
    Iapp_I_uAcm2=0.0,
    alpha_n_per_ms=0.5,   # Eq. 9 parameter a_n (not given numerically in main text)
):
    start_scope()
    np.random.seed(rng_seed)
    seed(rng_seed)
    defaultclock.dt = dt_ms * ms

    # Convert knobs to SI
    gNI_loc = gNI_mS_cm2 * MS_PER_CM2_TO_SI
    gEI_loc = gEI_mS_cm2 * MS_PER_CM2_TO_SI
    gNE_loc = gNE_mS_cm2 * MS_PER_CM2_TO_SI
    gEE_loc = gEE_mS_cm2 * MS_PER_CM2_TO_SI
    gIE_loc = gIE_mS_cm2 * MS_PER_CM2_TO_SI
    gII_loc = gII_mS_cm2 * MS_PER_CM2_TO_SI

    IappE_loc = Iapp_E_uAcm2 * UA_PER_CM2_TO_SI
    IappI_loc = Iapp_I_uAcm2 * UA_PER_CM2_TO_SI

    alpha_n = alpha_n_per_ms / ms

    # Put all external symbols in a namespace for clarity/robustness
    ns = dict(
        # conductances / drives
        gNI_loc=gNI_loc, gEI_loc=gEI_loc, gNE_loc=gNE_loc,
        gEE_loc=gEE_loc, gIE_loc=gIE_loc, gII_loc=gII_loc,
        IappE_loc=IappE_loc, IappI_loc=IappI_loc,
        # constants
        EL=EL, VT_E=VT_E, VT_I=VT_I, VR=VR, V_peak=V_peak,
        Vex=Vex, Vin=Vin, V_K=V_K, Mg=Mg,
        C_E=C_E, C_I=C_I, gL_E=gL_E, gL_I=gL_I,
        a_adapt=a_adapt, d_adapt=d_adapt,
        sigma_E=sigma_E, sigma_I=sigma_I,
        tau_n=tau_n, tau_e=tau_e, tau_ei=tau_ei, tau_i=tau_i,
        alpha_n=alpha_n
    )

    eqs_E = """
    I_intrinsic = gL_E * (v - EL) * (v - VT_E) / (VT_E - EL) : amp/meter**2

    B = 1.0 / (1.0 + (Mg * exp(-0.062 * (v/mV)) / 3.57)) : 1

    I_AMPA = gEE_loc * se_EE * (v - Vex) : amp/meter**2
    I_NMDA = gNE_loc * sn_EE * B * (v - Vex) : amp/meter**2
    I_GABA = gIE_loc * si_IE * (v - Vin) : amp/meter**2
    I_syn  = I_AMPA + I_NMDA + I_GABA : amp/meter**2

    I_adapt = z * (v - V_K) : amp/meter**2

    dv/dt = (IappE_loc + I_intrinsic - I_adapt - I_syn + sigma_E*xi) / C_E : volt

    dz/dt = -a_adapt * z : siemens/meter**2

    se_EE : 1
    sn_EE : 1
    si_IE : 1
    """

    eqs_I = """
    I_intrinsic = gL_I * (v - EL) * (v - VT_I) / (VT_I - EL) : amp/meter**2

    B = 1.0 / (1.0 + (Mg * exp(-0.062 * (v/mV)) / 3.57)) : 1

    I_AMPA = gEI_loc * se_EI * (v - Vex) : amp/meter**2
    I_NMDA = gNI_loc * sn_EI * B * (v - Vex) : amp/meter**2
    I_GABA = gII_loc * si_II * (v - Vin) : amp/meter**2
    I_syn  = I_AMPA + I_NMDA + I_GABA : amp/meter**2

    dv/dt = (IappI_loc + I_intrinsic - I_syn + sigma_I*xi) / C_I : volt

    se_EI : 1
    sn_EI : 1
    si_II : 1
    """

    E = NeuronGroup(N_E, eqs_E, threshold="v >= V_peak",
                    reset="v = VR; z += d_adapt",
                    method="euler", namespace=ns, name="E")

    I = NeuronGroup(N_I, eqs_I, threshold="v >= V_peak",
                    reset="v = VR",
                    method="euler", namespace=ns, name="I")

    E.v = EL + 1*mV * randn(N_E)
    I.v = EL + 1*mV * randn(N_I)
    E.z = 0 * siemens/meter**2

    # --- Synapses ---


    S_EE = Synapses(
        E, E,
        model="""
        dse/dt = -se/tau_e : 1 (clock-driven)
        dsn/dt = alpha_n*se*(1-sn) - sn/tau_n : 1 (clock-driven)
        se_EE_post = se : 1 (summed)
        sn_EE_post = sn : 1 (summed)
        """,
        on_pre="se += 1.0",
        namespace=ns,
        name="S_EE"
    )
    S_EE.connect(condition="i != j", p=P_EE)
    S_EE.delay = 0.5*ms

    S_EI = Synapses(
        E, I,
        model="""
        dse/dt = -se/tau_ei : 1 (clock-driven)
        dsn/dt = alpha_n*se*(1-sn) - sn/tau_n : 1 (clock-driven)
        se_EI_post = se : 1 (summed)
        sn_EI_post = sn : 1 (summed)
        """,
        on_pre="se += 1.0",
        namespace=ns,
        name="S_EI"
    )
    S_EI.connect(p=P_EI)
    S_EI.delay = 0.5*ms

    S_IE = Synapses(
        I, E,
        model="""
        dsi/dt = -si/tau_i : 1 (clock-driven)
        si_IE_post = si : 1 (summed)
        """,
        on_pre="si += 1.0",
        namespace=ns,
        name="S_IE"
    )
    S_IE.connect(p=P_IE)
    S_IE.delay = 0.5*ms

    S_II = Synapses(
        I, I,
        model="""
        dsi/dt = -si/tau_i : 1 (clock-driven)
        si_II_post = si : 1 (summed)
        """,
        on_pre="si += 1.0",
        namespace=ns,
        name="S_II"
    )
    S_II.connect(condition="i != j", p=P_II)
    S_II.delay = 0.5*ms

    # --- Monitors ---
    spE = SpikeMonitor(E)
    spI = SpikeMonitor(I)
        # --- LFP recording without storing all voltages ---
    lfp_list = []
    t_list   = []

    @network_operation(dt=defaultclock.dt)
    def record_lfp():
        lfp_list.append(float(np.mean(E.v / mV)))
        t_list.append(float(defaultclock.t / second))

    run(T_ms * ms)

    lfp   = np.asarray(lfp_list)
    t_lfp = np.asarray(t_list)

    rate_E = (spE.num_spikes / N_E) / ((T_ms*ms)/second)
    rate_I = (spI.num_spikes / N_I) / ((T_ms*ms)/second)

    return dict(
        lfp=lfp,
        t_lfp=t_lfp,
        spE_t=np.asarray(spE.t/second), spE_i=np.asarray(spE.i),
        spI_t=np.asarray(spI.t/second), spI_i=np.asarray(spI.i),
        rate_E=float(rate_E), rate_I=float(rate_I),
        params=dict(alpha_n_per_ms=float(alpha_n_per_ms),
                    gNI_mS_cm2=float(gNI_mS_cm2),
                    dt_ms=float(dt_ms), T_ms=float(T_ms))
    )



    