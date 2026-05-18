#include "../include/mpo.hpp"
#include "../include/mps.hpp"
int main(int argc, char ** argv)
{
    MPO<D> op = MPO<D>::load(argv[1]);
    MPS<Z> s1 = MPS<Z>::load(argv[2]);
    cout << "READ" << endl;
    s1 = multiply(op,s1);
    s1.save(argv[3]);
    return 0;
}