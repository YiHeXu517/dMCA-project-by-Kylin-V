import numpy as np
from numpy.linalg import norm,svd
from scipy.constants import c,h,e

S = 0.5
wavenum_2_eV = c * 100.0 * (h / e)
e_s1 = 19940.0 * wavenum_2_eV
v12 = -211.0 * wavenum_2_eV
wvib = 1150.0 * wavenum_2_eV
rabi = 0.3
N = 6
g = rabi / 2.0 / np.sqrt(2*N)

def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x - mu)**2) / (2 * sigma**2))
def Jw(omega,dw):
    lam = 0.2
    gam = 0.01
    res = lam * omega * gam / (np.power(omega,2) + gam**2)
    return res

Nvib = 1000
windows = [0.01,0.41]
wx = np.linspace(windows[0],windows[1],Nvib)
dvib = (windows[1] - windows[0]) / Nvib
gvib = np.zeros((Nvib,))
for vib in range(Nvib-1):
    gvib[vib] = np.sqrt( dvib * 0.5 * (Jw(wx[vib],dvib) + Jw(wx[vib+1],dvib)) )
    if vib>0 and wx[vib-1]<wvib and wx[vib]>wvib:
        gvib[vib] = -np.sqrt(2.0 * S) * wvib

def lanczos( Nbk ):
    ex_vib = norm(gvib)
    Krys = [gvib/ex_vib]
    ''' lanczos '''
    for k in range(Nbk):
        w = wx * Krys[k]
        for j in range(k+1):
            mu = Krys[j].T@w
            w -= Krys[j] * mu
        Krys.append(w/norm(w))

    Xi = np.vstack(Krys)
    Xi , R = np.linalg.qr(Xi.T,mode="reduced")
    xHx = Xi.T@(np.diag(wx)@Xi)
    ex_vib *= Krys[0]@Xi[:,0]
    return xHx,ex_vib
Nbk = 75
xHx,G = lanczos(Nbk)
print(G)

lattice = ["cav"] + ["{}e{}".format(_+1,x) for _ in range(N) for x in (1, 2)]
phys = [3] + [2 for i in range(2*N)]

for k in range(Nbk):
    lattice = lattice + ["{}e{}v{}".format(_+1,x,k+1) for _ in range(N) for x in (1, 2)]
    phys = phys + [8 for i in range(2*N)]

lattice_ref = dict()
for k in range(len(lattice)):
    lattice_ref[lattice[k]] = k

e_cav = 2.43
Terms = [("cav","N",e_cav)]
for e in range(N):
    Terms.append( ("{}e1".format(e+1),"N",e_s1) )
    Terms.append( ("{}e2".format(e+1),"N",e_s1) )
    Terms.append( ("{}e1".format(e+1),"cav","+","-",g) )
    Terms.append( ("{}e2".format(e+1),"cav","+","-",g) )
    Terms.append( ("{}e1".format(e+1),"cav","-","+",g) )
    Terms.append( ("{}e2".format(e+1),"cav","-","+",g) )
    Terms.append( ("{}e1".format(e+1),"{}e2".format(e+1),"+","-",v12) )
    Terms.append( ("{}e1".format(e+1),"{}e2".format(e+1),"-","+",v12) )
    Terms.append( ("{}e1".format(e+1),"{}e1v1".format(e+1),"N","Q",G) )
    Terms.append( ("{}e2".format(e+1),"{}e2v1".format(e+1),"N","Q",G) )
    for v in range(Nbk):
        Terms.append( ("{}e1v{}".format(e+1,v+1),"N",xHx[v,v]) )
        Terms.append( ("{}e2v{}".format(e+1,v+1),"N",xHx[v,v]) )
    for v in range(Nbk-1):
        Terms.append( ("{}e1v{}".format(e+1,v+1),"{}e1v{}".format(e+1,v+2),"+","-",xHx[v,v+1]) )
        Terms.append( ("{}e2v{}".format(e+1,v+1),"{}e2v{}".format(e+1,v+2),"+","-",xHx[v,v+1]) )
        Terms.append( ("{}e1v{}".format(e+1,v+1),"{}e1v{}".format(e+1,v+2),"-","+",xHx[v+1,v]) )
        Terms.append( ("{}e2v{}".format(e+1,v+1),"{}e2v{}".format(e+1,v+2),"-","+",xHx[v+1,v]) )

print(len(lattice))
f1 = open("mero_dimer.inp",'w')
print(phys.__str__()[1:-1].replace(',',' '),file=f1)
for tm in Terms:
    for tml in range(len(tm)):
        if tml < (len(tm) - 1)//2:
            print("{} ".format(lattice_ref[tm[tml]]),end='',file=f1)
        else:
            print("{} ".format(tm[tml]),end='',file=f1)
    print("\n",end='',file=f1)
f1.close()
