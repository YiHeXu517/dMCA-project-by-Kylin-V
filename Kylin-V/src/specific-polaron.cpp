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
    for(int ext=0;ext<13;++ext)
    {
        for(int vib=13;vib<25;++vib)
        {
            vector<char> polaron(lat.nsites,'I');
            polaron[ext] = '+';
            polaron[vib] = '+';
            MPO<D> o1(lat.gen_global_op(polaron));
            MPS<Z> s1(multiply(o1,vacuum));
            Dense<Z,1> corr({6000});
            for(int step=0;step<num_steps;++step)
            {
                corr({step}) = overlap(s1,ests[step]); 
            }
            string fncr = "State/correlation_" + to_string(ext) + "_" + to_string(vib);
            ofstream ofs(fncr,std::ios::binary);
            ofs.write(reinterpret_cast<char*>(corr.ptr),corr.size*sizeof(Z));
            ofs.close();
        }
            vector<char> polaron(lat.nsites,'I');
            polaron[ext] = '+';
            MPO<D> o1(lat.gen_global_op(polaron));
            MPS<Z> s1(multiply(o1,vacuum));
            Dense<Z,1> corr({6000});
            for(int step=0;step<num_steps;++step)
            {
                corr({step}) = overlap(s1,ests[step]); 
            }
            string fncr = "State/correlation_" + to_string(ext);
            ofstream ofs(fncr,std::ios::binary);
            ofs.write(reinterpret_cast<char*>(corr.ptr),corr.size*sizeof(Z));
            ofs.close();
 
    }

    return 0;
}
