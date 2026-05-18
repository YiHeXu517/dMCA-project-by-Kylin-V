#include "../include/tdvp.hpp"
#include "../include/lattice.hpp"
int main(int argc, char ** argv)
{
    MPO<D> ham = MPO<D>::load("Op/Ham");
    MPO<D> c_num = MPO<D>::load("Op/cavity_num");
    MPS<Z> vacuum = MPS<Z>::load("State/vac.mps");
    int num_steps = 6000;
    double dt =  0.25;
    vector<MPS<Z>> ests(num_steps);
    for(int step=0;step<num_steps;++step)
    {
        string fn = "State/Es_" + to_string(step) + ".mps";
        ests[step] = MPS<Z>::load(fn.c_str());
    }
    // build lef 
    Lattice lat(argv[1]);
    vector<char> exciton(lat.nsites,'I'), small_polaron(lat.nsites,'I'), medium_polaron(lat.nsites,'I'),
    large_polaron(lat.nsites,'I'), photon_polaron(lat.nsites,'I');
    exciton[1] = '+';
    small_polaron[1] = '+';
    small_polaron[13]  = '+';
    medium_polaron[1] = '+';
    medium_polaron[14] = '+';
    large_polaron[1] = '+';
    large_polaron[15] = '+';
    photon_polaron[0] = '+';
    photon_polaron[13]  = '+';
    MPO<D> o1(lat.gen_global_op(exciton)), 
    o113(lat.gen_global_op(small_polaron)),
    o114(lat.gen_global_op(medium_polaron)),
    o115(lat.gen_global_op(large_polaron)),
    o013(lat.gen_global_op(photon_polaron));
    MPS<Z> s1(multiply(o1,vacuum)),
    s113(multiply(o113,vacuum)),
    s114(multiply(o114,vacuum)),
    s115(multiply(o115,vacuum)),
    s013(multiply(o013,vacuum)); 

    // overlap
    ofstream ofs("correlation");
    for(int step=0;step<num_steps;++step)
    {
        Z ov = overlap(ests[0],ests[step]),
        ov1 = overlap(s1,ests[step]),
        ov113 = overlap(s113,ests[step]),
        ov114 = overlap(s114,ests[step]),
        ov115 = overlap(s115,ests[step]),
        ov013 = overlap(s013,ests[step]),
        pop = expectation(ests[step],c_num,ests[step]);
        ofs << std::scientific << setprecision(8)
        << real(ov) << " " << imag(ov) << " " 
        << real(ov1) << " " << imag(ov1) << " " 
        << real(ov113) << " " << imag(ov113) << " "
        << real(ov114) << " " << imag(ov114) << " "
        << real(ov115) << " " << imag(ov115) << " "
        << real(ov013) << " " << imag(ov013) << " "
        << real(pop) << " " << imag(pop) << endl;
    }
    ofs.close();
    return 0;
}
