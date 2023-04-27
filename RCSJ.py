'''
Definitions and basic functions for the RCSJ model and associated calculations.
'''
import trevorarp as tb
from trevorarp.physics import phi0 # Magnetic Flux Quantum in Wb = V*s
import matplotlib.pyplot as plt
import warnings
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

def solve_RCSJ(i, phia, betaC, betaL, y0=None, alphaI=0, alphaR=0, alphaC=0, iN1=0, iN2=0, tdelta=0.2, Npts=200):
    '''
    Solve the RCSJ using the normalized (Lavangin type) equations, where we solve for time averaged voltages.
    Quantities are variables are defined in the SQUID Handbook chapter 2.2.2

    Args:
        i : Normalized bias current i = I/I0 for average critical current I0 = (I0_1 + I0_2)/2
        phia : Applied flux
        betaC : Hysteresis parameter
        betaL : Inductance parameter
        y0 : The inital condition of the simulation. If None auto-generates.
        alphaI, alphaR, alphaC : The asymmetry parameters for I, R and C. Zero by default
        iN1, iN2 : The noise current parameters for both junctions. Zero by default.
        tdelta : The time point spacing in units of time constant tau for simulation
        Npts : Number of points to time average over.
        retOutput : If true returns the full output of the simulation, not just the final result.

    Returns:
        (t, wsolved) : Time and the full output of the simulation.
    '''

    def dydt(w,t):
        d1, y1, d2, y2 = w

        j = (2/betaL)*((d2-d1)/(2*np.pi) - phia)
        a1 = -1.0 * (0.5 * i + j) + (1-alphaI) * np.sin(d1) + iN1
        b1 = 1 - alphaR
        c1 = betaC * (1 - alphaC)
        a2 = -1.0 * (0.5 * i - j) + (1 + alphaI) * np.sin(d2) + iN2
        b2 = 1 + alphaR
        c2 = betaC * (1 + alphaC)
        f = [y1,
             -1.0*(b1/c1)*y1 - (a1/c1),
             y2,
             -1.0*(b2/c2)*y2 - (a2/c2)]
        return f

    tpts = np.arange(0,tdelta*Npts, tdelta)

    if y0 is None:
        # Define the inital condition, Seems to converge after a few time units.
        y0 = [0, 0, 2*np.pi*phia, 0] # Initial conditions, normally converges regardless
    wsolved = odeint(dydt, y0, tpts)
    return tpts, wsolved

def Sophisticated_calc_timeaveraged_RCSJ(i, phia, betaC, betaL, alphaI=0, alphaR=0, alphaC=0, iN1=0, iN2=0, Nperiods=6, sim_length=200, max_iter=10, start_len=100):
    '''
    AN ATTEMPT TO DO A MORE SOPHISTICATED WAY OF TIME AVERAGING. WAS MORE EFFICIENT BUT THE RESULTS HAD LOTS OF ARTIFACTS.
    
    Performs the RCSJ simulation efficiently and time averages the Josephson voltage consistently.

    Solve the RCSJ using the normalized (Lavangin type) equations, where we solve for time averaged voltages.
    Quantities are variables are defined in the SQUID Handbook chapter 2.2.2

    Args:
        i : Normalized bias current i = I/I0 for average critical current I0 = (I0_1 + I0_2)/2
        phia : Applied flux
        betaC : Hysteresis parameter
        betaL : Inductance parameter
        alphaI, alphaR, alphaC : The asymmetry parameters for I, R and C. Zero by default
        iN1, iN2 : The noise current parameters for both junctions. Zero by default.
        Nperiods : The number of periods to average over, will simulate until has that number of periods.
        sim_length : The number of points to simulate on each iteration
        max_iter : Will break out after this number of iterations
        start_len : Number of iterations ot start up from inital condition

    Returns:
        v: the appropriately time averaged dimensionless voltage across junction 1, obtained from delta' via the josephson relation.
            Is usually equal to voltage across junction 2.
    '''

    # Initial run up of the simulation
    t_, y_ = solve_RCSJ(i, phia, betaC, betaL, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC, iN1=iN1, iN2=iN2, Npts=start_len)
    y0 = y_[start_len-1,:]

    cnt = 1
    t, y = solve_RCSJ(i, phia, betaC, betaL, y0=y0, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC,
                                       iN1=iN1, iN2=iN2, Npts=sim_length)
    while cnt < max_iter:
        if np.abs(np.max(y_[:,1])-np.min(y_[:,1])) < 1e-5: # If it's not varying significantly with time, probably in SC state
            return np.mean(y_[:,1])
        ind = np.argwhere(np.diff(np.sign(y[:,1]-np.mean(y[:,1])))>0)
        if len(ind) >= Nperiods: # Make sure you have sufficient minima to compute. If not keep going
            N = len(ind)
            ix1 = 0
            if N%2 == 0: # Make sure that you take an even number of crossings.
                ix2 = N
            else:
                ix2 = N - 1
            return np.mean(y[ix1:ix2,1])
        t_, y_ = solve_RCSJ(i, phia, betaC, betaL, y0=y0, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC, iN1=iN1, iN2=iN2,
                            Npts=sim_length)
        y = np.append(y, y_, axis=0)
        cnt += 1
    print("Warning calc_timeaveraged_RCSJ: Maximum number of iterations exceeded value may not be accurate.")
    return np.mean(y[:, 1]) # If you exceed maximum iterations

def calc_timeaveraged_RCSJ(i, phia, betaC, betaL, alphaI=0, alphaR=0, alphaC=0, iN1=0, iN2=0, sim_length=1000, start_len=100):
    '''
    Performs the RCSJ simulation efficiently and time averages the Josephson voltage consistently.

    Solve the RCSJ using the normalized (Lavangin type) equations, where we solve for time averaged voltages.
    Quantities are variables are defined in the SQUID Handbook chapter 2.2.2

    Time averaged voltage taken in brute force manner, i.e. just averaged after running the model for a certain amount.

    Args:
        i : Normalized bias current i = I/I0 for average critical current I0 = (I0_1 + I0_2)/2
        phia : Applied flux
        betaC : Hysteresis parameter
        betaL : Inductance parameter
        alphaI, alphaR, alphaC : The asymmetry parameters for I, R and C. Zero by default
        iN1, iN2 : The noise current parameters for both junctions. Zero by default..
        sim_length : The number of points to simulate
        start_len : Number of iterations ot start up from inital condition

    Returns:
        v: the voltage across the SQUID. Obtained by averaging together the time averaged values of the voltages across
            both junctions. Generically the time averaged voltages of the two junctions should be the same.
    '''

    # Initial run up of the simulation
    t_, y_ = solve_RCSJ(i, phia, betaC, betaL, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC, iN1=iN1, iN2=iN2, Npts=start_len)
    y0 = y_[start_len-1,:]

    t, y = solve_RCSJ(i, phia, betaC, betaL, y0=y0, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC,
                                       iN1=iN1, iN2=iN2, Npts=sim_length)
    return (np.mean(y[:, 1])+np.mean(y[:, 3]))/2

def solve_SQUID(ibias, B, icritical=50e-6, Rshunt=2.9, diameter=100e-9, R=1.0, betaC=1, betaL=1, Rpara=0, alphaI=0, alphaR=0, alphaC=0, iN1=0, iN2=0):
    '''
    Solve the RCSJ in a load-line circuit where there is a shunt resistor and the current into the Tip is not an independent
    varible, but is deterbined by a current
    Quantities are variables are defined in the SQUID Handbook chapter 2.2.2.

    Args:
        ibias : Normalized bias current in A.
        B : The external magnetic field (Tesla)
        diameter: The effective diameter of the SQUID
        icritical: The critical current of the SQUID (averaged over both junctions I0 = (I0_1 + I0_2)/2 ).
        Rshunt: The resistance of the shunt resistor in
        R : The resistance value in the RCSJ model. Engineer with SHOVET.
        phia : Applied flux
        betaC : Hysteresis parameter
        betaL : Inductance parameter
        Rpara : The parasitic resistance of the Tip.
        y0 : The inital condition of the simulation. If None auto-generates.
        alphaI, alphaR, alphaC : The asymmetry parameters for I, R and C. Zero by default
        iN1, iN2 : The noise current parameters for both junctions. Zero by default.
        tdelta : The time point spacing in units of time constant tau for simulation
        Npts : Number of points to time average over.
        retOutput : If true returns the full output of the simulation, not just the final result.

    Returns:
        Itip the current going through the tip
    '''
    warnings.filterwarnings('ignore')

    phia = B*(np.pi*(diameter/2)**2) # Put field in units of flux and normalize by the flux quantum.
    phia = phia/phi0
    def func(x):
        itip = x[0]
        if itip == 0:
            itip = 1e-9
        v = calc_timeaveraged_RCSJ(itip/icritical, phia, betaC, betaL, alphaI=alphaI, alphaR=alphaR, alphaC=alphaC, iN1=iN1, iN2=iN2)
        V = icritical*R*v
        Rtip = V/itip + Rpara
        diff = itip - (Rshunt/(Rtip + Rshunt))*ibias
        return diff
    x = fsolve(func, np.array([ibias/2]))
    return x[0] # Current through the Tip

def solve_SQUID_realunits(ibias, icritical, Rshunt, phia, R, C, L, alphaI=0, alphaR=0, alphaC=0, iN1=0, iN2=0):
    '''
    Similar to solve_SQUID but uses real units L and C instead of betaC and betaL
    '''
    betaC = (2*np.pi/phi0)*icritical*R*R*C
    betaL = 2*L*icritical/phi0
    return solve_SQUID(ibias, icritical, Rshunt, phia, R, betaC, betaL, alphaI, alphaR, alphaC, iN1, iN2)
#

def gen_SQUID_iden(Icrit, diameter, Rs, Rpara):
    '''
    Autogenerates a name for a SQUID with the given parameters.
    '''
    return str(int(Icrit/1e-6)) + str(int(diameter/1e-9)) + str(int(10*Rs)) + str(int(10*Rpara))

def gen_SIM_iden(Ibias, B, rows, cols, R, betaC, betaL):
    '''
    Autogenerates a name for a SQUID simulation from parameters
    '''
    s =  str(rows) + str(cols) + str(int(R*10)) + str(int(betaC*1000)) + str(int(betaL*1000)) + "-"
    s = s + str(int(np.min(Ibias / 1e-6))) + str(int(np.max(Ibias / 1e-6))) + str(int(np.min(B / 1e-3))) + str(int(np.max(B / 1e-3)))
    return s


if __name__ == "__main__":
    # Performance Testing
    import time

    Icrit = 100e-6
    diameter = 150e-9
    Rs = 2.5  # Ohms
    R = 2.0  # Ohms
    Rpara = 1.0  # Ohms parasitic resistance
    betaC = 0.1
    betaL = 0.5

    Ibias = 190e-6 #194e-6
    B = 45e-3 #41e-3

    phia = B * (np.pi * (diameter / 2) ** 2)  # Put field in units of flux and normalize by the flux quantum.
    phia = phia / phi0

    t0 = time.time()
    Itip = solve_SQUID(Ibias, B, Icrit, Rs, diameter, R, betaC, betaL, Rpara)
    t1 = time.time()
    print(round(t1 - t0, 3), Itip)

    N = 1000
    t_, d_ = solve_RCSJ(Itip/Icrit, phia, betaC, betaL, tdelta=0.2, Npts=100)
    t, d = solve_RCSJ(Itip/Icrit, phia, betaC, betaL, tdelta=0.2, Npts=N, y0=d_[49, :])
    fi = tb.display.figure_inches(None, "1", "1", dark=True)
    ax1 = fi.make_axes()
    rows, cols = d.shape
    ax1.plot(t, d[:, 1], color='C0', label=r'$\delta_{1}^{\prime}$')
    ax1.plot(t, d[:, 3], color='C1', label=r'$\delta_{2}^{\prime}$')
    ax1.axhline(np.mean(d[:, 1]), color='C0', ls='--')
    ax1.axhline(np.mean(d[:, 3]), color='C1', ls='--')

    ax1.set_xlabel(r'time (units of $\tau$)')
    ax1.legend()
    plt.show()