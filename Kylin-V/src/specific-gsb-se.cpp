#include "../include/tdvp.hpp"
#include "../include/lattice.hpp"
int main(int argc, char ** argv)
{
    Lattice lat(argv[1]);
    MPO<D> c_down = MPO<D>::load("Op/cavity_down");
    size_t Nt = 6000;
    double dt = 0.25;
    vector<MPO<Z>> propagators(Nt);
    vector<MPS<Z>> gstates(Nt);
    for(size_t t=0;t<Nt;++t)
    {
        propagators[t] = lat.gen_diag_exponential(-1.0*t*dt);
        string fn = "State/Es_" + to_string(t) + ".mps";
        gstates[t] = multiply(c_down,MPS<Z>::load(fn.c_str()));
    }
    size_t Nt_nl = 600;
    size_t T2s[4] = {1200,1500,1800,2100};
    for(size_t t2=0;t2<4;++t2)
    {
        Dense<Z,2> gsb({Nt_nl,Nt_nl}),se({Nt_nl,Nt_nl});
        for(size_t t1=0;t1<Nt_nl;++t1)
        {
            cout << "T1 at " << t1 << endl;
            for(size_t t3=0;t3<Nt_nl;++t3)
            {
                gsb({t1,t3}) = expectation(gstates[t1],propagators[T2s[t2]+t3],gstates[t3]);
                se({t1,t3}) = expectation(gstates[T2s[t2]+t1],propagators[t3],gstates[T2s[t2]+t3]);
            }
        }
        string fg = "GSB_R_" + to_string(T2s[t2]) + ".cout", fs = "SE_R_" + to_string(T2s[t2]) + ".cout";
        ofstream ofg(fg,std::ios::binary),ofs(fs,std::ios::binary);
        ofg.write(reinterpret_cast<char*>(gsb.ptr),gsb.size*sizeof(Z));
        ofs.write(reinterpret_cast<char*>(se.ptr),se.size*sizeof(Z));
        ofg.close();
        ofs.close();
    }
    return 0;
}
