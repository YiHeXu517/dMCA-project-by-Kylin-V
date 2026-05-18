#include "../include/lattice.hpp"
#include "../include/mps.hpp"
#include "getopt.h"
#include "unistd.h"
int main(int argc, char ** argv)
{
    if(access("Op",F_OK)==-1) { int status = mkdir("Op",S_IRWXU | S_IRWXG | S_IROTH |S_IXOTH); 
    if(status!=0) { cout << "Fail to create New Dir" << endl; exit(1); } }
    if(access("State",F_OK)==-1) { int status = mkdir("State",S_IRWXU | S_IRWXG | S_IROTH |S_IXOTH); 
    if(status!=0) { cout << "Fail to create New Dir" << endl; exit(1); } }
    Lattice lat(argv[1]);
    MPO<D> ham = lat.gen_total_para();
    ham.print();
    ham.save("Op/Ham");
    vector<char> op_cav_up(lat.nsites,'I'),op_cav_down(lat.nsites,'I'),op_cav_num(lat.nsites,'I');
    op_cav_up[0] = '+';
    op_cav_down[0] = '-';
    op_cav_num[0] = 'N';
    MPO<D> cav_up(lat.gen_global_op(op_cav_up)), cav_down(lat.gen_global_op(op_cav_down)), cav_num(lat.gen_global_op(op_cav_num));
    cav_up.save("Op/cavity_up");
    cav_down.save("Op/cavity_down");
    cav_num.save("Op/cavity_num");
    MPS<Z> vacuum(lat.nsites);
    for(int s=0;s<lat.nsites;++s)
    {
        Dense<Z,3> ri({lat.nphysdims[s],1,1});
        ri.ptr[0] = 1.0;
        vacuum.tt[s] = move(ri);
    }
    vacuum.save("State/vac.mps");
    return 0;
}
