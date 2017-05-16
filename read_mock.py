# 02 May 2017 00:48:45
import h5py
import numpy as np
import matplotlib.pyplot as plt
try:
    from zypy.zycosmo import get_halofuncs, get_cosmo, CosmoParams
    has_hmf = True
except ImportError:
    has_hmf = False
    print("Cosmology/HMF code is required.")

try:
    from hod.shmr import SHMR_Leauthaud, HSMR, HSMR_split
    from hod.predict import get_f_sigma_lnMs
    has_shmr = True
except ImportError:
    print("SHMR code is required.")
    has_shmr = False


def _read_recdict_from_hdf5(h5file):
    """Read a dict of record arrays from hdf5."""
    f = h5py.File(h5file, "r")
    recdict = {}
    for grp, val in f.iteritems():
        print grp
        datasets = []
        dtypes = []
        for key in f[grp].keys():
            dset = f[grp][key][:]
            dtypename = f[grp][key].dtype.name
            dtype = (str(key), dtypename)
            datasets.append(dset)
            dtypes.append(dtype)
        print dtypes
        recdict[str(grp)] = np.rec.fromarrays(tuple(datasets), dtype=dtypes)
    f.close()
    return(recdict)

def read_mock(mockfile):
    """Read hdf5 galaxy mock data into a numpy.recarray"""
    recdict = _read_recdict_from_hdf5(mockfile)
    mockdata = recdict['galaxy']
    print "The data columns are: "
    print mockdata.dtype.names
    return(mockdata)

def read_mock_hmf(mockfile, mmin=1.e9, mmax=1.e16, nmbin=101, h=0.701, rcube=250.0):
    """Read Halo Mass Function.

    Returns
    ---
    Mh_arr: ndarray
        Halo mass in Msun.

    dndlnMh_arr: ndarray
        Number density in # per lnMsun per Mpc^3.
    """
    galrec = read_mock(mockfile)
    iscen = galrec['lg_halo_mass'] > 0
    Mh_arr = np.logspace(np.log10(mmin), np.log10(mmax), nmbin)
    wid = np.log10(Mh_arr[1] / Mh_arr[0])
    _wid = np.log(Mh_arr[1] / Mh_arr[0])
    # print wid
    _Mh_arr = np.zeros(Mh_arr.size+1)
    _Mh_arr[1:] = Mh_arr * 10**(0.5*wid)
    _Mh_arr[0] = Mh_arr[0] / 10**(0.5*wid)
    dn_arr = np.histogram(galrec['lg_halo_mass'][iscen] - np.log10(h), bins=np.log10(_Mh_arr))[0]
    # dndMh_arr = dn_arr / (_Mh_arr[1:] - _Mh_arr[:-1])
    dndlnMh_arr = dn_arr / _wid
    vol = (rcube / h)**3
    dndlnMh_arr /= vol
    return(Mh_arr, dndlnMh_arr)

def test_mock_hmf(mockfile, rcube=250.0):
    """Compare the halo mass function in the mock to theory prediction."""
    Mh_arr, dndlnMh_arr = read_mock_hmf(mockfile, mmin=1.e10, mmax=1.e16, nmbin=50, h=0.7, rcube=rcube)
    plt.plot(np.log10(Mh_arr), dndlnMh_arr, 'k-', label="Simulation")
    if has_hmf:
        # compare with theory
        # Bolshoi cosmology
        cosmo = CosmoParams(omega_M_0=0.27, sigma_8=0.82, h=0.70, omega_b_0=0.0469, n=0.95, set_flat=True)
        _Mh_arr, _dndMh_arr = get_halofuncs(z=0.1, cosmo=cosmo, DELTA_HALO=200.0, mmin=1.e10, mmax=1.e16, nmbin=50)[:2]
        _dndlnMh_arr = _dndMh_arr * _Mh_arr
        plt.plot(np.log10(_Mh_arr), _dndlnMh_arr, 'r--', label="Theory")
        # plt.plot(np.log10(_Mh_arr), dndlnMh_arr/_dndlnMh_arr, 'k-', label="Simulation/Theory")
        # plt.axhline(1)
    plt.legend(loc=1)
    plt.yscale('log')
    plt.xlabel(r"$M_h\;[M_\odot]$")
    plt.ylabel(r"$dn/d\ln M_h$")
    # plt.ylim(0.9, 1.1)
    plt.ylim(1e-8, 1e-1)
    plt.show()

def test_mock_shmr(mockfile):
    """Check the stellar to halo mass relation in the mock."""
    galrec = read_mock(mockfile)
    iscen = galrec['lg_halo_mass'] > 1
    lgmh = galrec['lg_halo_mass'][iscen]
    lgms = galrec['lg_stellar_mass'][iscen]
    lgmh_bins = np.linspace(11.4, 15.0, 35)
    lgmh_cens = (lgmh_bins[1:] + lgmh_bins[:-1]) / 2.0
    lgms_cens = np.empty_like(lgmh_cens)
    lgms_scas = np.empty_like(lgmh_cens)
    for i in xrange(lgmh_cens.size):
        sel = (lgmh >= lgmh_bins[i]) & (lgmh < lgmh_bins[i+1])
        lgms_cens[i] = np.mean(lgms[sel])
        lgms_scas[i] = np.std(lgms[sel])
    if has_shmr:
        lgMs_0 = 10.30790
        lgMh_1 = 12.09899
        beta = 0.33272
        delta = 0.440
        gamma = 1.20579
        Mh_1 = 10**lgMh_1  # Msun rather than Msun/h
        Ms_0 = 10**lgMs_0  # this is always Msun/h^2
        shmr = SHMR_Leauthaud(Mh_1=Mh_1, Ms_0=Ms_0, beta=beta, delta=delta, gamma=gamma)
        # scatter in the SHMR
        sigma_lnMs = 0.49730
        eta = -0.04104
        lgMh_sca = lgMh_1
        Mh_sca = 10**lgMh_sca
        f_sigma_lnMs = get_f_sigma_lnMs(sigma_lnMs=sigma_lnMs, eta=eta, Mh_sca=Mh_sca)
        #
        mharr = 10**lgmh_cens  / 0.701
        lnmsarr = shmr.log_stellarmass_mean(np.log(mharr))
        lgmsarr = lnmsarr / np.log(10.0)
        lgmssca = f_sigma_lnMs(mharr) / np.log(10.0)
        plt.plot(lgmh_cens, lgmsarr, 'r-', label="Theory")
        plt.plot(lgmh_cens, lgmsarr + lgmssca , 'r--')
        plt.plot(lgmh_cens, lgmsarr - lgmssca , 'r--')
    plt.plot(lgmh_cens, lgms_cens, 'k-', label="Simulation")
    plt.plot(lgmh_cens, lgms_cens+lgms_scas, 'k--')
    plt.plot(lgmh_cens, lgms_cens-lgms_scas, 'k--')
    plt.legend(loc=2)
    plt.xlabel(r"$M_h\;[M_\odot/h]$")
    plt.ylabel(r"$M_*\;[M_\odot/h^2]$")
    plt.show()

def test_mock_hsmr(mockfile):
    """Check the halo to stellar mass relation in the mock."""
    galrec = read_mock(mockfile)
    iscen = galrec['lg_halo_mass'] > 1
    lgmh = galrec['lg_halo_mass'][iscen]
    lgms = galrec['lg_stellar_mass'][iscen]
    lgms_bins = np.linspace(9.8, 12.0, 31)
    lgms_cens = (lgms_bins[1:] + lgms_bins[:-1]) / 2.0
    lgmh_cens = np.empty_like(lgms_cens)
    lgmh_scas = np.empty_like(lgms_cens)
    for i in xrange(lgms_cens.size):
        sel = (lgms >= lgms_bins[i]) & (lgms < lgms_bins[i+1])
        lgmh_cens[i] = np.mean(lgmh[sel])
        lgmh_scas[i] = np.std(lgmh[sel])
    if has_shmr:
        lgMs_0 = 10.30790
        lgMh_1 = 12.09899
        beta = 0.33272
        delta = 0.440
        gamma = 1.20579
        Mh_1 = 10**lgMh_1  # Msun rather than Msun/h
        Ms_0 = 10**lgMs_0  # this is always Msun/h^2
        shmr = SHMR_Leauthaud(Mh_1=Mh_1, Ms_0=Ms_0, beta=beta, delta=delta, gamma=gamma)
        # scatter in the SHMR
        sigma_lnMs = 0.49730
        eta = -0.04104
        lgMh_sca = lgMh_1
        Mh_sca = 10**lgMh_sca
        f_sigma_lnMs = get_f_sigma_lnMs(sigma_lnMs=sigma_lnMs, eta=eta, Mh_sca=Mh_sca)
        #
        cosmo = CosmoParams(omega_M_0=0.27, sigma_8=0.82, h=0.70, omega_b_0=0.0469, n=0.95, set_flat=True)
        # M_arr, dndM_arr = get_halofuncs(z=0.1, cosmo=cosmo, DELTA_HALO=200.0, mmin=1.e9, mmax=1.e16, nmbin=300)[:2]
        M_arr, dndM_arr = get_halofuncs(z=0.1, cosmo=cosmo, DELTA_HALO=200.0, mmin=1.e10, mmax=1.e16, nmbin=240)[:2]
        if True:
            lnMh_arr = np.log(M_arr)
            dndlnMh_arr = dndM_arr * M_arr
            np.savetxt('hmf.dat', np.vstack((lnMh_arr, dndlnMh_arr)).T)
        # HSMR
        hsmr = HSMR(shmr, f_sigma_lnMs, dndM_arr, M_arr, lgmsmin=8.0, lgmsmax=13.0, dlgms=0.02)
        lnMh_mean, lnMh_mean2, lnMh_med, sigma_lnMh_low, sigma_lnMh_upp = hsmr.get_plnMh_at_lnMs()
        #
        h = 0.70
        lgms_arr = hsmr.lnMs_arr / np.log(10.0)
        lgmharr = lnMh_mean / np.log(10.0) + np.log10(h)
        # lgmharr = lnMh_med / np.log(10.0) + np.log10(h)
        sigupp = sigma_lnMh_upp / np.log(10.0)
        siglow = sigma_lnMh_low / np.log(10.0)
        plt.plot(lgms_arr, lgmharr, 'r-', label="Theory")
        plt.plot(lgms_arr, lgmharr + sigupp , 'r--')
        plt.plot(lgms_arr, lgmharr - siglow , 'r--')
        #
    plt.plot(lgms_cens, lgmh_cens, 'k-', label="Simulation")
    plt.plot(lgms_cens, lgmh_cens+lgmh_scas, 'k--')
    plt.plot(lgms_cens, lgmh_cens-lgmh_scas, 'k--')
    plt.legend(loc=2)
    plt.xlabel(r"$M_*\;[M_\odot/h^2]$")
    plt.ylabel(r"$M_h\;[M_\odot/h]$")
    plt.show()


if __name__ == "__main__":
    mockfile = '/Users/ying/Data/ihodmock/standard/iHODcatalog_bolshoi.h5'
    # read_mock(mockfile)
    # test_mock_hmf(mockfile)
    # test_mock_shmr(mockfile)
    test_mock_hsmr(mockfile)
