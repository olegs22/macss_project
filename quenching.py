import numpy as np
import emcee
from prob2d_mine import get_prob2d1
from read_mock import read_mock
mock = 'iHODcatalog_bolshoi.h5'

def exp_Mh_split(theta, use_red):

    prob2_sp, probm_sp, probh_sp = get_prob2d1(theta, use_red)

    #int_sp = np.trapz(prob2_sp, x = probm_sp, axis=0)
    #norm_sp = np.trapz(int_sp, x=probh_sp)
    #print np.abs(norm_sp - 1.0) > 1e-5

    p_ms_sp = np.trapz(prob2_sp, probh_sp, axis=-1)
    #print np.trapz(p_ms_sp,probm_sp)

    joint_sp = []
    for i in range(len(probm_sp)):
        joint_sp.append(prob2_sp[i,:] / p_ms_sp[i])


    joint_sp = np.array(joint_sp)

    for j in range(len(probm_sp)):
        norm2s = np.trapz(joint_sp[j,:],probh_sp)
        #if np.abs(norm2s - 1.) > 1e-5:
        #print 'error'

    exp_sp = []
    for k in range(len(probm_sp)):
        exp_sp.append(np.trapz(joint_sp[k,:]*probh_sp, probh_sp))
    exp_sp = np.array(exp_sp)
    lgexp = (exp_sp/np.log(10.)) + np.log10(0.7)
    lgprobm = probm_sp / np.log(10.)

    return lgprobm, lgexp

def isred1(log_smass, gcolor1):
    cut = 0.8*(log_smass/10.5)**0.6
    isred2 = gcolor1 >= cut
    return(isred2)

def comp_splot(use_red):
    galrec = read_mock(mock)
    iscen = galrec['lg_halo_mass'] > 1
    lgmh1 = galrec['lg_halo_mass'][iscen]
    lgms1 = galrec['lg_stellar_mass'][iscen]
    gcolor1 = galrec['g-r'][iscen]
    isred_m = isred1(lgms1, gcolor1)
    # check which of them are red vs blue
    if use_red:
        colsel = isred_m
        xlabel = r"$M_*^{\mathrm{red}}\;[M_\odot/h^2]$"
        color = 'r'
        label = 'Mock Red Data'
        _label = 'Predicted Red'
    else:
        colsel = ~isred_m
        xlabel = r"$M_*^{\mathrm{blue}}\;[M_\odot/h^2]$"
        color = 'b'
        label = 'Mock Blue Data'
        _label = 'Predicted Blue'
    # prediction!
    #model
    #lgMs_arr, lgMh_mean_arr = exp_Mh_split(theta, use_red)
    #plt.plot(lgMs_arr, lgMh_mean_arr, color=color, label=_label)
    # select red or blue centrals
    lgmh1 = lgmh1[colsel]
    lgms1 = lgms1[colsel]
    # do measurements in the same way
    lgms_bins = np.linspace(9.5, 11., 122)
    lgms_cens = (lgms_bins[1:] + lgms_bins[:-1]) * 0.5
    lgmh_cens = np.zeros_like(lgms_cens)
    lgmh_errs = np.zeros_like(lgms_cens)
    for i in xrange(lgms_cens.size):
        sel = (lgms1 >= lgms_bins[i]) & (lgms1 < lgms_bins[i+1])
        nsel = np.sum(sel)
        if nsel > 5:
            # update lgms_cens
            lgms_cens[i] = np.mean(lgms1[sel])
            lgmh_cens[i] = np.mean(lgmh1[sel])
            lgmh_errs[i] = np.std(lgmh1[sel]) / np.sqrt(float(nsel))
    return lgms_cens, lgmh_cens, lgmh_errs

data_ms, data_mh, data_err = comp_splot(use_red=False)

def lnlike(theta):
    model_ms, model_mh = exp_Mh_split(theta,use_red=False)

    p = (((data_mh - model_mh) / data_err)**2) + np.log(2. * np.pi * data_err**2)

    return -0.5*np.sum(p)

def prior(theta):
    mh_p, mu = theta
    if 0.001 < mu <= 1.0:
        return 0.0
    return -np.inf

def ln_post(theta):
    P = prior(theta)
    if not np.isfinite(P):
        return -np.inf
    return P + lnlike(theta)
ndim = 2
nwalkers = 200
z = np.zeros((ndim,nwalkers))
init_pos = (9.,0.5)
h = 1e-1
pos_i =[]
for i in range(ndim):
    z[i,:] = init_pos[i] + h*np.random.randn(nwalkers)

for i in range(nwalkers):
    pos_i.append(np.array([z[0,i],z[1,i]]))

b_steps, steps = 100, 500

sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post, threads=10)

pos,prob,state = sampler.run_mcmc(pos_i, b_steps)
print sampler.acceptance_fraction.mean()
sampler.reset()
_,_,_ = sampler.run_mcmc(pos, steps,rstate0=state)
print sampler.acceptance_fraction.mean()

np.savetxt('chains.txt', sampler.flatchain)
