import numpy as np
from numpy.fft import fft,fftshift,fftfreq,ifft

Nt_nl = 600
files = ["SE_R_1200.cout","SE_R_1500.cout","SE_R_1800.cout","SE_R_2100.cout","SE_R_540.cout"]
gamma = np.ones(100) * 0.01
for fn in files:
    gsb = np.fromfile(fn,dtype=np.complex128,count=Nt_nl*Nt_nl)
    gsb = np.reshape(gsb,(Nt_nl,Nt_nl))

    om1,om3,steps = np.linspace(2,3,100),np.linspace(2,3,100),np.arange(Nt_nl)*0.25

    U1 = np.exp(-1.0j * om1[:,None] * steps[None,:] - gamma[:,None] * steps[None,:])
    U3 = np.exp(1.0j  * steps[:,None] * om3[None,:] - steps[:,None] * gamma[None,:])

    gsb = U1@gsb@U3

    np.save(fn+".npy",gsb)
