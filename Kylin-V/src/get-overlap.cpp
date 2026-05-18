#include "../include/mpo.hpp"
#include "../include/mps.hpp"
int main(int argc, char ** argv)
{
    MPS<Z> s1 = MPS<Z>::load(argv[1]);
    MPS<Z> s2 = MPS<Z>::load(argv[2]);
    cout << std::scientific << setprecision(8) << overlap(s1,s2) << endl;
    return 0;
}