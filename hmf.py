# 10 May 2017 10:00:56
import numpy as np
import matplotlib.pyplot as plt


def load_hmf(hmfdata='../data/hmf.dat'):
    """Load Halo Mass Function from hmfdata."""
    lnMh_arr, dndlnMh_arr = np.genfromtxt(hmfdata, unpack=True)
    return(lnMh_arr, dndlnMh_arr)


if __name__ == "__main__":
    lnMh_arr, dndlnMh_arr = load_hmf()
    # convert ln to lg for plotting purposes
    lgMh_arr = lnMh_arr / np.log(10.0)
    plt.plot(lgMh_arr, dndlnMh_arr, 'r-')
    plt.yscale('log')
    plt.xlabel(r'$\lg\;M_h\;[M_{\odot}]$')
    plt.ylabel(r'$dn/\ln\;M_h$')
    plt.show()


