import numpy as np
from numpy.fft import fft,fftshift,fftfreq,ifft

def fft_abs(ct,gam=0.01):
    dt = 0.25
    C = np.conj(ct[1:])[::-1]
    C = np.concatenate([C, ct])
    Nt = ct.shape[0]
    decay_t = np.exp(-np.abs(gam*np.arange(-Nt+1,Nt)*dt))
    cw = fftshift(ifft(C*decay_t))
    return cw


def nonzero_indices_same_mod12(nums):

    indices = [i for i, x in enumerate(nums) if x != 0]
    if len(indices) <= 1:
        return indices, True
    target = indices[0] % 12
    is_same = all((i % 12) == target for i in indices[1:])
    return is_same


Nb,Nt = 78977,11999
ph_basis,ex_basis,ph_vib_basis,ex_s_vib_basis,ex_l_vib_basis = [],[],[],[],[]
ph_ov,ex_ov,ph_vib_ov,ex_s_vib_ov,ex_l_vib_ov = np.zeros((Nt,),dtype=complex),np.zeros((Nt,),dtype=complex),np.zeros((Nt,),dtype=complex),np.zeros((Nt,),dtype=complex),np.zeros((Nt,),dtype=complex)
ds = dict()
f1 = open("all_basis")
ss = f1.readline()
while ss:
    ss = f1.readline()
    if len(ss)>1:
        lis1 = ss.split(":")
        lis2 = lis1[-1][1:-2].split(",")
        for s2 in range(len(lis2)):
            lis2[s2] = int(lis2[s2])
        if lis2[0]==1:
            if lis2[1:] == [0] * (len(lis2)-1):
                ph_basis.append(int(lis1[0]))
            else:
                ph_vib_basis.append(int(lis1[0]))
        else:
            if lis2[13:] == [0] * (len(lis2)-13):
                ex_basis.append(int(lis1[0]))
            else:
                if nonzero_indices_same_mod12(lis2):
                    ex_s_vib_basis.append(int(lis1[0]))
                else:
                    ex_l_vib_basis.append(int(lis1[0]))

        ds[tuple(lis2)] = int(lis1[0])

f1.close()
Nt = 6000
print(ph_basis,ph_vib_basis,ex_basis,ex_s_vib_basis,ex_l_vib_basis)
jut = np.fromfile("State/overlaps",dtype=np.complex128,count=Nb*Nt).reshape((Nb,Nt))
jut[:,1:] = jut[:,:-1]
jut[:,0] *= 0
jut[ph_basis[0],0] = 1.0

Absw = np.abs(fft_abs(jut[ph_basis[0],:],0.0))

for pvb in ph_vib_basis:
    fju = fft_abs(jut[pvb,:],0.0)
    ph_vib_ov += fju * fju.conj() / Absw

for eb in ex_basis:
    fju = fft_abs(jut[eb,:],0.0)
    ex_ov += fju * fju.conj() / Absw

for evb in ex_s_vib_basis:
    fju = fft_abs(jut[evb,:],0.0)
    ex_s_vib_ov += fju * fju.conj() / Absw

for evb in ex_l_vib_basis:
    fju = fft_abs(jut[evb,:],0.0)
    ex_l_vib_ov += fju * fju.conj() / Absw

Nt = 6000
dt = 0.25

np.save("Abs-broaden",np.abs(fft_abs(jut[ph_basis[0],:],0.01)))
np.save("Ex-ov",ex_ov.real)
np.save("Pol-s-ov",ex_s_vib_ov.real)
np.save("Pol-l-ov",ex_l_vib_ov.real)
np.save("Phot-Phon-ov",ph_vib_ov.real)
np.save("Abs-narrow",Absw.real)
