#include "../include/tdvp.hpp"
int main(int argc, char ** argv)
{
    MPO<D> op = MPO<D>::load(argv[1]);
    MPS<Z> s1 = MPS<Z>::load(argv[2]);
    int num_steps = stoi(string(argv[3]));
    double dt =  stod(string(argv[4]));

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
    int switch_to_single = 50;
    for(int step=0;step<num_steps;++step)
    {
        cout << "Step " << step+1 << endl;
        string fn = "State/Es_" + to_string(step) + ".mps";
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
            driver.state_.save(fn.c_str());
        }
        else
        {
            driver.impl_single(dt,1e-8,16);
            auto[max_site,max_bond] = driver.state_.max_bond();
            cout << "Maximal bond dimension = " << max_bond << " on site " << max_site << endl;
            driver.state_.save(fn.c_str());
        }

    }
    return 0;
}
