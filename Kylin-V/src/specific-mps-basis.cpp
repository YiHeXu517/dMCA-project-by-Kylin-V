#include "../include/tdvp.hpp"
#include "../include/lattice.hpp"
tuple<vector<vector<int>>,vector<Dense<Z,2>>> dominant_mps(MPS<Z> const & s0, double tol)
{
    int ns = s0.tt.size();
    vector<Dense<Z,2>> env(1);
    Dense<Z,2> env_b({1,1});
    env_b.ptr[0] = 1.0;
    env[0] = move(env_b);
    vector<vector<int>> env_basis(1);
    for(int s=0;s<ns;++s)
    {
        int nd = s0.tt[s].shape[0];
        int nv = env.size();
        vector<Dense<Z,2>> env_cur;
        vector<vector<int>> env_basis_cur;
        int nr = env[0].shape[0];
        int nc = s0.tt[s].shape[2];
        int nk = s0.tt[s].shape[1];
        Z ones(1.0), zeros(0.0);
        for(int v=0;v<nv;++v)
        {
            for(int d=0;d<nd;++d)
            {
                Dense<Z,2> m2({nr,nc});
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                nr, nc, nk, &ones,
                env[v].ptr, nk, 
                s0.tt[s].ptr+d*nk*nc, nc, 
                &zeros, m2.ptr, nc);
                if(m2.norm()>=tol)
                {
                    env_cur.push_back(m2);
                    vector<int> m2_basis(env_basis[v]);
                    m2_basis.push_back(d);
                    env_basis_cur.push_back(m2_basis);
                }
            }
        }
        swap(env,env_cur);
        swap(env_basis,env_basis_cur);
    }
    return make_tuple(env_basis,env);
}
Z ci_overlap(MPS<Z> const & s0, vector<int> const & c0)
{
    int ns = c0.size();
    Dense<Z,2> env({1,1});
    env.ptr[0] = 1.0;
    for(int s=0;s<ns;++s)
    {
        int nr = env.shape[0];
        int nc = s0.tt[s].shape[2];
        int nk = s0.tt[s].shape[1];
        Z ones(1.0), zeros(0.0);
        Dense<Z,2> m2({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                nr, nc, nk, &ones,
                env.ptr, nk, 
                s0.tt[s].ptr+c0[s]*nk*nc, nc, 
                &zeros, m2.ptr, nc);
        env = move(m2);
    }
    return env.ptr[0];
}
int main(int argc, char ** argv)
{
    int num_steps = 6000;
    double dt =  0.25;
    vector<MPS<Z>> ests(num_steps);
    double tol = 5e-4;

    for(int step=0;step<num_steps;++step)
    {
        string fn = "State/Es_" + to_string(step) + ".mps";
        ests[step] = MPS<Z>::load(fn.c_str());

    }
    auto[basis,coef] = dominant_mps(ests.back(),tol);
    int num_basis = basis.size();
    cout << "Total " << num_basis << "basis." << endl;
    Dense<Z,2> ovs({num_basis,num_steps});
    for(int b=0;b<num_basis;++b)
    {
       cout << b << ":" << basis[b] << endl;
       for(int t=0;t<num_steps;++t)
       {
           ovs({b,t}) = ci_overlap(ests[t],basis[b]);
       }
    }
    string fncr = "State/overlaps";
    ofstream ofs(fncr,std::ios::binary);
    ofs.write(reinterpret_cast<char*>(ovs.ptr),ovs.size*sizeof(Z));
    ofs.close();
 
    return 0;
}
