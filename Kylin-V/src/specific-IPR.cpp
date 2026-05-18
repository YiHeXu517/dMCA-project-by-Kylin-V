#include "../include/tdvp.hpp"
#include "../include/lattice.hpp"
Dense<Z,4> mps_to_mpo(Dense<Z,3> const & s)
{
    int nd = s.shape[0], nl = s.shape[1], nr = s.shape[2];
    Dense<Z,4> res({nd,nd,nl,nr});
    for(int d=0;d<nd;++d)
    {
        for(int l=0;l<nl;++l)
        {
            for(int r=0;r<nr;++r)
            {
                res({d,d,l,r}) = s({d,l,r});
            }
        }
    }
    return res;
}
Dense<Z,4> dense_apply_mpo_mps_site(Dense<Z,3> const & envl, Dense<Z,3> const & mps, Dense<Z,4> const & mpo)
{
    Dense<Z,3> s2 = transpose<Z,3>(mps,{1,0,2});
    Dense<Z,4> ls2 = mm<Z,3,3,1>(envl,s2);
    ls2 = transpose<Z,4>(ls2,{0,3,1,2});
    Dense<Z,4> opt2 = transpose<Z,4>(mpo,{2,1,0,3});
    opt2 = mm<Z,4,4,2>(ls2,opt2);
    opt2 = transpose<Z,4>(opt2,{2,0,3,1});
    return opt2;
}
MPS<Z> dense_apply_mpo_mps(vector<Dense<Z,4>> const & op, MPS<Z> const & s0)
{
    int ns = s0.tt.size();
    Dense<Z,3> env({1,1,1});
    env.ptr[0] = 1.0;
    MPS<Z> res(ns);
    for(int s=0;s<ns;++s)
    {
        Dense<Z,4> cc = dense_apply_mpo_mps_site(env,s0.tt[s],op[s]);
        if(s!=ns-1)
        {
            auto[lef,rig] = svd<Z,2,2>(cc,'r');
            res.tt[s] = move(lef);
            env = move(rig);
        }
        else
        {
            Dense<Z,3> rr({cc.shape[0],cc.shape[1],1});
            memcpy(rr.ptr,cc.ptr,cc.size*sizeof(Z));
            res.tt[ns-1] = move(rr);
        }
    }
    res.canon();
    return res;
}
Z get_IPR(MPS<Z> const & s0)
{
    int ns = s0.tt.size();
    vector<Dense<Z,4>> op0(ns);
    for(int s=0;s<ns;++s)
    {
        op0[s] = mps_to_mpo(s0.tt[s]);
    }
    MPS<Z> s2 = dense_apply_mpo_mps(op0,s0);
    return overlap(s2,s2);
}
int main(int argc, char ** argv)
{
    int num_steps = 6000;
    double dt =  0.25;
    for(int step=0;step<num_steps;step=step+60)
    {
        string fn = "State/Es_" + to_string(step) + ".mps";
        MPS<Z> est = MPS<Z>::load(fn.c_str());
        cout << std::scientific << setprecision(8) << real(get_IPR(est)) << endl;
    }
    return 0;
}
