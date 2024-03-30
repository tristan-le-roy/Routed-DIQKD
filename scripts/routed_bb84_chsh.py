"""
Script to compute converging lower bounds on the DIQKD rates of a routed Bell protocol using
devices that are constrained by some 2 input 2 output distribution.
More specifically, computes a sequence of lower bounds on the problem

            inf H(A|x=0,E) - H(A|x=0, y=2, B)

where the infimum is over all quantum devices with some expected behaviour. 

Code also analyzes the scenario where we have inefficient detectors.
Noisy-preprocessing can be used in order to boost rates. 

The protocols we consider are as follows:
    Alice measures X and Z.
    Closeby Bob measures Z+X and Z-X.
    Distant Bob measures one of the two cases:
        1. Z+X, Z-X, X. Key is generated when Alice and Bob both measure X (i.e., x = 0 and y = 2).
        2. X and Z. Key is generated when Alice and Bob both measure X (i.e., x = 0 and y = 0).

All the parties bin their no-click outcomes except for the distant Bob, who does not bin his outcomes only for the key generating input.
Note that when BobL measures X, if the round is a test round then he bins but if it is a key round then he does not bin.
"""

def cond_ent(joint, marg):
    """
    Returns H(A|B) = H(AB) - H(B)

    Inputs:
        joint    --     joint distribution on AB
        marg     --     marginal distribution on B
    """

    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob*log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob*log2(prob)

    return hab - hb

def objective(ti, q):
    """
    Returns the objective function for the faster computations.
        Key generation on X=0
        Only two outcomes for Alice

        ti     --    i-th node
        q      --    bit flip probability
    """
    obj = 0.0
    # F = [A[0][0], 1 - A[0][0]]                # POVM for Alices key gen measurement
    F = [P([0],[0],"A"), P([1],[0],"A")]   
    for a in range(A_config[0]):
        b = (a + 1) % 2                     # (a + 1 mod 2)
        M = (1-q) * F[a] + q * F[b]         # Noisy preprocessing povm element
        obj += M * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def compute_entropy(SDP, q, momentequality_constraints):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP   --   sdp relaxation object
        q     --   probability of bitflip
    """
    
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    # with it. Added a nontrivial bound when removing the final term
    # (WARNING: proof is not yet in the associated paper).
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 2 * q * (1-q) * W[-1] / log(2)

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], q)
        if T[k]!=1:
            a_i = 3/2 * max(1/T[k] , 1/(1-T[k]))
            moment_inequality_constraints = [a_i-z*Dagger(z) for z in Z]+[a_i-Dagger(z)*z for z in Z]
        else:
            moment_inequality_constraints = []

        SDP.set_objective(new_objective)
        SDP.process_constraints(momentequalities=momentequality_constraints,momentinequalities=moment_inequality_constraints)
        # SDP.solve('mosek', solverparameters = {'num_threads': int(NUM_SUBWORKERS)})
        SDP.solve()

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            print('Bad solve: ', k, SDP.status)
            break

    return ent

def HAgB(MA,rho, etaA, etaBL, q):
    """
    Computes the error correction term in the key rate for a given system,
    a fixed detection efficiency and noisy preprocessing. Computes the relevant
    components of the distribution and then evaluates the conditional entropy.

        eta    --    detection efficiency
        q      --    bitflip probability
    """
    id = qtp.qeye(2)[:]
    # Noiseless measurements
    a00 = MA[0,0]
    b20 = MA[0,0] #Bob's measurement for Y=2 is the same as Alice's for X=0

    # Alice bins to 0 transforms povm
    A00 = etaA * a00 + (1-etaA) * id
    # Final povm transformation from the bitflip
    A00 = (1-q) * A00 + q * (id - A00)
    A01 = id - A00

    if BobL_bins_for_key_generation == False:
        # Bob has inefficient measurement but doesn't bin
        B20 = etaBL * b20 
        B21 = etaBL * (id - b20)
        B22 = (1-etaBL) * id

        # joint distribution
        q00 = np.real(np.trace(rho @ np.kron(A00,B20))) # (rho*qtp.tensor(A00, B20)).tr().real
        q01 = np.real(np.trace(rho @ np.kron(A00,B21)))#(rho*qtp.tensor(A00, B21)).tr().real
        q02 = np.real(np.trace(rho @ np.kron(A00,B22)))#(rho*qtp.tensor(A00, B22)).tr().real
        q10 = np.real(np.trace(rho @ np.kron(A01,B20)))#(rho*qtp.tensor(A01, B20)).tr().real
        q11 = np.real(np.trace(rho @ np.kron(A01,B21)))#(rho*qtp.tensor(A01, B21)).tr().real
        q12 = np.real(np.trace(rho @ np.kron(A01,B22)))#(rho*qtp.tensor(A01, B22)).tr().real

        qb0 = np.real(np.trace(rho @ np.kron(id,B20)))#(rho*qtp.tensor(id, B20)).tr().real
        qb1 = np.real(np.trace(rho @ np.kron(id,B21)))#(rho*qtp.tensor(id, B21)).tr().real
        qb2 = np.real(np.trace(rho @ np.kron(id,B22)))#(rho*qtp.tensor(id, B22)).tr().real

        qjoint = [q00,q01,q02,q10,q11,q12]
        qmarg = [qb0,qb1,qb2]

        return cond_ent(qjoint, qmarg)
    else:
        # Bob has inefficient measurement and bins
        B20 = etaBL * b20 + (1-etaBL) * id
        B21 = id - B20

        # joint distribution
        q00 = np.real(np.trace(rho @ np.kron(A00,B20)))
        q01 = np.real(np.trace(rho @ np.kron(A00,B21)))
        q10 = np.real(np.trace(rho @ np.kron(A01,B20)))
        q11 = np.real(np.trace(rho @ np.kron(A01,B21)))

        qb0 = np.real(np.trace(rho @ np.kron(id,B20)))
        qb1 = np.real(np.trace(rho @ np.kron(id,B21)))

        qjoint = [q00,q01,q10,q11]
        qmarg = [qb0,qb1]

        return cond_ent(qjoint, qmarg)

def get_qm_implementation():
    v = visibility
    [I, X, Y, Z] = [qtp.qeye(2)[:], qtp.sigmax()[:], qtp.sigmay()[:], qtp.sigmaz()[:]]
    [XR , ZR] = [(X+Z)/sqrt(2) , (X-Z)/sqrt(2)]

    phip=qtp.bell_state('00')[:] @ qtp.bell_state('00')[:].T.conj()
    phip = v*phip + (1-v)*qtp.qeye(4)[:] / 4
    MA=np.zeros((nX,nA), dtype=object)
    MB=np.zeros((nY,nB), dtype=object)

    [MA[0,0], MA[1,0], MA[0,1], MA[1,1]] = [(I+X)/2 ,(I-X)/2, (I+Z)/2, (I-Z)/2]  
    [MB[0,0], MB[1,0], MB[0,1], MB[1,1]] = [(I+XR)/2 ,(I-XR)/2, (I+ZR)/2, (I-ZR)/2]

    return phip, MA, MB

def get_p_ideal(phip,MA,MB):
    pchsh = np.zeros((nA,nB,nX,nY))
    pbb84= np.zeros((nA,nB,nX,nY))
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    pchsh[a,b,x,y] = np.real(np.trace(phip @ np.kron(MA[a,x],MB[b,y])))
                    pbb84[a,b,x,y] = np.real(np.trace(phip @ np.kron(MA[a,x],MA[b,y])))

    #pbb84 = pchsh ## Comment this line if you want BB84 correlations in the long path. Uncomment if you want chsh correlations in the long path. 

    return pchsh, pbb84

def get_p_obs(pchsh,pbb84,etaA,etaBS,etaBL,Alice_bins_for_test_rounds,BobS_bins_for_test_rounds,BobL_bins_for_test_rounds): 
    nZ = 2
    p_obs = np.zeros((nA+1,nB+1,nX,4,nZ) , dtype=object)
    """
    Add the CHSH in short path and BB84 in long path
    """
    ### Add the probabilities for 'click' events
    for a in range(nA):
        for b in range(nB):
            for x in range(nX):
                for y in range(nY):
                    p_obs[a,b,x,y,0] += etaA * etaBS * pchsh[a,b,x,y]
                    p_obs[a,b,x,y,1] += etaA * etaBL * pbb84[a,b,x,y]

                    if Alice_bins_for_test_rounds == True:
                        p_obs[a,b,x,y,0] += (1 - etaA) * etaBS * np.sum(pchsh[:,b,0, y]) * (1-a)
                        p_obs[a,b,x,y,1] += (1 - etaA) * etaBL * np.sum(pbb84[:,b,0, y]) * (1-a)
                    if BobS_bins_for_test_rounds == True:
                        p_obs[a,b,x,y,0] += etaA * (1 - etaBS) * np.sum(pchsh[a,:, x,0]) * (1-b)
                    if BobL_bins_for_test_rounds == True:
                        p_obs[a,b,x,y,1] += etaA * (1 - etaBL) * np.sum(pbb84[a,:, x,0]) * (1-b)
                    if Alice_bins_for_test_rounds == True and BobS_bins_for_test_rounds == True:
                        p_obs[a,b,x,y,0] += (1 - etaA) * (1 - etaBS)*(1-a)*(1-b)
                    if Alice_bins_for_test_rounds == True and BobL_bins_for_test_rounds == True:
                        p_obs[a,b,x,y,1] += (1 - etaA) * (1 - etaBL)*(1-a)*(1-b)

    ### Add the probabilities for 'no-click' events
    
    for x in range(nX):
        for y in range(nY):
            if Alice_bins_for_test_rounds == False and BobS_bins_for_test_rounds == False:
                p_obs[2,2,x,y,0] += (1 - etaA) * (1 - etaBS)
            if Alice_bins_for_test_rounds == False and BobL_bins_for_test_rounds == False:
                p_obs[2,2,x,y,1] += (1 - etaA) * (1 - etaBL)

            for a in range(nA):
                if BobS_bins_for_test_rounds == False:
                    p_obs[a,2,x,y,0] += etaA * (1-etaBS) * np.sum(pchsh[a,:,x,0])
                if BobL_bins_for_test_rounds == False:
                    p_obs[a,2,x,y,1] += etaA * (1-etaBL) * np.sum(pbb84[a,:,x,0])
                if Alice_bins_for_test_rounds == True and BobS_bins_for_test_rounds == False:
                    p_obs[a,2,x,y,0] += (1 - etaA) * (1 - etaBS)*(1-a)
                if Alice_bins_for_test_rounds == True and BobL_bins_for_test_rounds == False:
                    p_obs[a,2,x,y,1] += (1 - etaA) * (1 - etaBL)*(1-a)
            for b in range(nB):
                if Alice_bins_for_test_rounds == False:
                    p_obs[2,b,x,y,0] += (1-etaA) * etaBS * np.sum(pchsh[:,b,0,y])
                    p_obs[2,b,x,y,1] += (1-etaA) * etaBL * np.sum(pbb84[:,b,0,y])
                if Alice_bins_for_test_rounds == False and BobS_bins_for_test_rounds == True:
                    p_obs[2,b,x,y,0] += (1 - etaA) * (1 - etaBS)*(1-b)
                if Alice_bins_for_test_rounds == False and BobL_bins_for_test_rounds == True:
                    p_obs[2,b,x,y,1] += (1 - etaA) * (1 - etaBL)*(1-b)



    """
    Add the CHSH in long path
    """
    for a in range(nA):
        for b in range(nB):
            for x in range(nX):
                for y in range(nY):
                    p_obs[a,b,x,y+2,1] += etaA * etaBL * pchsh[a,b,x,y]

                    if Alice_bins_for_test_rounds == True:
                        p_obs[a,b,x,y+2,1] += (1 - etaA) * etaBL * np.sum(pchsh[:,b,0, y]) * (1-a)
                    if BobL_bins_for_test_rounds == True:
                        p_obs[a,b,x,y+2,1] += etaA * (1 - etaBL) * np.sum(pchsh[a,:, x,0]) * (1-b)
                    if Alice_bins_for_test_rounds == True and BobL_bins_for_test_rounds == True:
                        p_obs[a,b,x,y+2,1] += (1 - etaA) * (1 - etaBL)*(1-a)*(1-b)

    ### Add the probabilities for 'no-click' events
    
    for x in range(nX):
        for y in range(nY):
            if Alice_bins_for_test_rounds == False and BobL_bins_for_test_rounds == False:
                p_obs[2,2,x,y+2,1] += (1 - etaA) * (1 - etaBL)

            for a in range(nA):
                if BobL_bins_for_test_rounds == False:
                    p_obs[a,2,x,y+2,1] += etaA * (1-etaBL) * np.sum(pchsh[a,:,x,0])
                if Alice_bins_for_test_rounds == True and BobL_bins_for_test_rounds == False:
                    p_obs[a,2,x,y+2,1] += (1 - etaA) * (1 - etaBL)*(1-a)
            for b in range(nB):
                if Alice_bins_for_test_rounds == False:
                    p_obs[2,b,x,y+2,1] += (1-etaA) * etaBL * np.sum(pchsh[:,b,0,y])
                if Alice_bins_for_test_rounds == False and BobL_bins_for_test_rounds == True:
                    p_obs[2,b,x,y+2,1] += (1 - etaA) * (1 - etaBL)*(1-b)
    
    
    return p_obs 

def score_constraints(P, p_obs):
    """
    Returns the moment equality constraints for the distribution. We only look at constraints coming
    from the inputs 0/1. Potential to improve by adding input 2 also?
    """
    constraints = []
    for a in range(nA):
        for b in range(nB):
            for x in range(nX):
                for y in range(nY):
                    constraints += [P([a,b],[x,y]) - p_obs[a, b, x, y, 0]]
                    constraints += [P([a,b],[x,y+2]) - p_obs[a, b, x, y, 1]]
                    constraints += [P([a,b],[x,y+4]) - p_obs[a, b, x, y+2, 1]]

    return constraints[:]

def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

         m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = P.substitutions
    # subs.update(ncp.projective_measurement_constraints(A,B))
    # Finally we note that Alice and distant BobL's (but not BobS's) operators should All commute with Eve's ops 
    Alice_ops = P.get_extra_monomials("A")
    BobL_ops = [P([j],[k],"B") for k in range(2,len(B_config)) for j in range(B_config[k]-1)]
    for a in Alice_ops+BobL_ops:
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})
    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = P.get_extra_monomials("A")
    Bflat = P.get_extra_monomials("B")
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]

    ###Add monos appearing in objective function
    for z in Z:
        monos += [P([0],[0],"A")*Dagger(z)*z]
    
    ##Add some more monos
    monos += [z*a*b for a in Aflat for b in Bflat[:2] for z in ZZ]
    # monos += [a*a_*b*z for a in Aflat for a_ in Aflat for b in Bflat for z in ZZ]
    # monos += [a*a_*z*b for a in Aflat for a_ in Aflat for b in Bflat[:2]  for z in ZZ]
    
    
    # monos += [a*b*bb*z for a in Aflat for b in Bflat for bb in Bflat for z in ZZ]
    # monos += [a*z*b*bb for a in Aflat for b in Bflat for bb in Bflat for z in ZZ]
    # monos += [a*b*z*bb for a in Aflat for b in Bflat for bb in Bflat for z in ZZ]
    return monos[:]

def get_levels_manually():
    """
    Returns the levels to use in the sdp relaxation. 
    Can be used to identify the monomials contributing most to the objective function.
    """
    ZZ = Z + [Dagger(z) for z in Z]
    # Aflat = P.get_extra_monomials("A")
    # Bflat = P.get_extra_monomials("B")
    level1 = ncp.flatten([P.get_all_operators(),ZZ]) 
    level2 = [u*v for u in level1 for v in level1]
    level3 = [u*v*w for u in level1 for v in level1 for w in level1]
    level4 = [u*v*w*x for u in level1 for v in level1 for w in level1 for x in level1]

    return level1+level2 

def get_keyrate_for_q(q, momentequality_constraints,etaBL):
    ent = compute_entropy(sdp,q,momentequality_constraints)
    err = HAgB(MA , phip, etaA, etaBL, q)
    keyrate = ent - err
    if noisy_preprocessing == True:
        return -keyrate    
    else:
        return ent, err, keyrate

def compute_keyrate(etaBL):
    p_obs = get_p_obs(pchsh, pbb84, etaA, etaBS, etaBL,Alice_bins_for_test_rounds,BobS_bins_for_test_rounds,BobL_bins_for_test_rounds) 
    momentequality_constraints = score_constraints(P, p_obs)
    if noisy_preprocessing == True:
        res = minimize_scalar(get_keyrate_for_q, bounds = (0,0.5),  args = (momentequality_constraints, etaBL), method='bounded', options={'disp': True})
        keyrate = -res.fun ; q = res.x
        return etaBL, keyrate, q
    else:
        ent, err, keyrate = get_keyrate_for_q(0, momentequality_constraints, etaBL); q = 0 ##without preprocessing
        return etaBL, keyrate, q, ent, err

"""
Now we start with setting up the ncpol2sdpa computations
"""
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from scipy.optimize import minimize_scalar
from sympy.physics.quantum.dagger import Dagger
import chaospy
import datetime



nA = 2; nB = 2; nX = 2; nY = 2
visibility = 1
BobL_bins_for_key_generation = False

Alice_bins_for_test_rounds = True
BobS_bins_for_test_rounds = True
BobL_bins_for_test_rounds = True

noisy_preprocessing = False ##optimize over noisy preprocessing
phip, MA, MB = get_qm_implementation()
pchsh, pbb84 = get_p_ideal(phip,MA,MB)

M = 6                           # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)      # Nodes, weights of quadrature
KEEP_M = False                      # Optimizing mth objective function?


# number of outputs for each inputs of Alice / Bobs devices
# (Dont need to include 3rd input for the distant Bob here as we only constrain the statistics
# for the other inputs).

if Alice_bins_for_test_rounds == True:
    A_config = [nA,nA]
else:
    A_config = [nA+1,nA+1]

if BobS_bins_for_test_rounds == True and BobL_bins_for_test_rounds == True:    
    B_config = [nB,nB,nB,nB,nB,nB] ##The first two are for BobS and the second four are for BobL
if BobS_bins_for_test_rounds == False and BobL_bins_for_test_rounds == True:
    B_config = [nB+1,nB+1,nB,nB,nB,nB]
if BobS_bins_for_test_rounds == True and BobL_bins_for_test_rounds == False:
    B_config = [nB,nB,nB+1,nB+1,nB+1,nB+1]
if BobS_bins_for_test_rounds == False and BobL_bins_for_test_rounds == False:
    B_config = [nB+1,nB+1,nB+1,nB+1,nB+1,nB+1]

P = ncp.Probability(A_config , B_config)
Z = ncp.generate_operators('Z', A_config[0], hermitian=False)

substitutions = get_subs()             # substitutions used in ncpol2sdpa
extra_monos = get_extra_monomials()    # extra monomials


ops = ncp.flatten([P.get_all_operators(),Z])        # Base monomials involved in problem

obj = objective(1,0)    # Placeholder objective function

p_obs = get_p_obs(pchsh,pbb84,1,1,1,Alice_bins_for_test_rounds,BobS_bins_for_test_rounds,BobL_bins_for_test_rounds) #placeholder p_obs
placeholder_moment_inequality_constraints = [T[0]-z*Dagger(z) for z in Z]+[T[0]-Dagger(z)*z for z in Z]
sdp = ncp.SdpRelaxation(ops, verbose = False, normalized=True)
sdp.get_relaxation(level = 2,
                    momentequalities = score_constraints(P, p_obs),
                    momentinequalities = placeholder_moment_inequality_constraints,
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

start_time = datetime.datetime.now(); print(start_time)

etaAs = [0.92,0.90,0.88]
etaAs.reverse()
for etaA in etaAs:
    etaBS = etaA; 
    etaBLs = np.arange(0.5,0.96,0.02).tolist() + np.linspace(0.96,0.99,6).tolist()[:-1] + np.linspace(0.99,0.999999,3).tolist()
    etaBLs.reverse()
    outputs = []
    for etaBL in etaBLs:
        try:
            etaBL_, keyrate_, q_, ent_, err_ = compute_keyrate(etaBL)
            outputs = outputs + [[etaBL_, keyrate_, q_, ent_, err_]]
            print("The key-rate for (etaA,etaBL) = ", (etaA,etaBL), " is ", keyrate_)
            if keyrate_ < 0:
                break
        except:
            print("Error in computing keyrate for etaA = ", etaA, "and etaBL = ", etaBL)
            pass
 
    ### Save the outputs to a file
    np.savetxt('chsh_bb84_bin_vis_1_etaA'+str(etaA)+'.txt', outputs)    

    print("etaS = ", etaA)
    print("I am Jef's cluster - computing bb84+chsh with binning, visibility = 1. Level 2+ABZ+AZB")
    end_time = datetime.datetime.now()
    print(end_time)
    

end_time = datetime.datetime.now()

print(end_time)

print("I am Jef's cluster - computing bb84+chsh with binning, visibility = 1. Level 2+ABZ+AZB")