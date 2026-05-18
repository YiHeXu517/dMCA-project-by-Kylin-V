#include "../include/mpo.hpp"
#include "../include/mps.hpp"
int main(int argc, char ** argv)
{
    MPS<Z> s1 = MPS<Z>::load(argv[1]);
    MPO<D> op = MPO<D>::load(argv[2]);
    MPS<Z> s2 = MPS<Z>::load(argv[3]);
    cout << std::scientific << setprecision(8) << expectation(s1,op,s2) << endl;
    return 0;
}