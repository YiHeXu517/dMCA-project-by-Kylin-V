# dMCA-project-by-Kylin-V

Steps:

1. run 'make' in Kylin-V by to get all binary files. You need C++ compiler and Intel MKL library.

2. run ../Kylin-V/bin/build-hamiltonian.a mero_dimer.inp in your path. (Build MPO/MPS)

3. run ../Kylin-V/bin/apply-op.a Op/cavity_up State/vac.mps a0.mps (Prepare initial state)

4. run ../Kylin-V/bin/time-evolution.a Op/Ham a0.mps 6000 0.25 (Time evolution)

5. run ../Kylin-V/bin/specific-linear.a mero_dimer.inp (Calculate linear absorption)

6. run ../Kylin-V/bin/specific-gsb-se.a mero_dimer.inp (Calculate GSB/SE response)

7. run  python3 -u make_2d_gsb_se.py (response to signal)

8. run ../Kylin-V/bin/specific-esa.a mero_dimer.inp (Calculate ESA response)

9. run python3 -u make_2d_esa.py (response to signal)

10. run ../Kylin-V/bin/specific-mps-tolerance.a (calculate cross correlations)

11. run python3 -u plot_com.py (check the results)
