#include "../include/tdvp.hpp"
#include "../include/lattice.hpp"
int main(int argc, char ** argv)
{
    MPO<D> op = MPO<D>::load("Op/Ham");
    MPO<D> c_up = MPO<D>::load("Op/cavity_up");
    size_t Nt = 6000;
    double dt = 0.25;
    vector<MPS<Z>> fstates(Nt);
    for(size_t t=0;t<Nt;++t)
    {
        string fn = "State/Es_" + to_string(t) + ".mps";
        fstates[t] = multiply(c_up,MPS<Z>::load(fn.c_str()));
    }
    size_t Nt_nl = 600;
    size_t T2s[4] = {1200,1500,1800,2100};
    for(size_t t2=0;t2<4;++t2)
    {
        MPS<Z> s1(fstates[T2s[t2]]);
        MPS<Z> hs1 = multiply(op,s1);
        hs1.tt[0] *= 0.0;
        s1 += hs1;

        size_t ns = s1.tt.size();
        for(size_t s=ns-1;s>0;--s)
        {
            Dense<Z,3> ttst = transpose<Z,3>(s1.tt[s],{1,0,2});
            auto[lef,rig] = svd<Z,1,2>(ttst,'l',1e-12);
            s1.tt[s] = transpose<Z,3>(rig,{1,0,2});
            s1.tt[s-1] = mm<Z,3,2,1>(s1.tt[s-1],lef);
        }

        TDVP driver(s1,op);
        int switch_to_single = 10;
        Dense<Z,2> esa({Nt_nl,Nt_nl});

        for(int step=0;step<Nt_nl;++step)
        {
            size_t t3 = (size_t)step;
            for(size_t t1=0;t1<Nt_nl;++t1)
            {
                esa({t1,t3}) = overlap(fstates[t1+T2s[t2]+t3],driver.state_);
            }
            cout << "Step " << step+1 << endl;
            if(step==switch_to_single)
            {
                driver.envr_.push_back(
                sweep(
                driver.envr_.back(),
                driver.state_.tt[1],
                driver.ham_.tt[1],
                driver.state_.tt[1],
                'l'));
            }
            if(step<switch_to_single)
            {
                driver.impl_double(dt,-1e-8,16);
                auto[max_site,max_bond] = driver.state_.max_bond();
                cout << "Maximal bond dimension = " << max_bond << " on site " << max_site << endl;
            }
            else
            {
                driver.impl_single(dt,1e-8,16);
                auto[max_site,max_bond] = driver.state_.max_bond();
                cout << "Maximal bond dimension = " << max_bond << " on site " << max_site << endl;
            }
        }
        string fe = "ESA_R_" + to_string(T2s[t2]) + ".cout";
        ofstream ofe(fe,std::ios::binary);
        ofe.write(reinterpret_cast<char*>(esa.ptr),esa.size*sizeof(Z));
        ofe.close();
    }
    return 0;
}
