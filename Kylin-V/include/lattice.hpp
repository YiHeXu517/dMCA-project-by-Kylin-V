#pragma once
#include "mpo.hpp"
#include "local.hpp"
using std::all_of;
using std::distance;
struct Lattice
{
    int nsites = 0;
    vector<int> nphysdims;
    vector<vector<char>> types;
    vector<double> rvals;
    vector<double> ivals;
    Lattice() = default;
    ~Lattice() = default;
    Lattice(const char * file_name)
    {
        int KTerm = 0;
        ifstream ifs(file_name);
        if(!ifs.is_open()) { cout << "Fail to open dump!" << endl; exit(1);}
        string lines,segs;
        getline(ifs,lines);
        stringstream ss_lines(lines);
        while(ss_lines >> segs)
        {
            if(segs.size()!=0)
            {
                nphysdims.push_back(stoi(segs));
            }
        }
        nsites = nphysdims.size();
        cout << "Total " << nsites << " sites." << endl;
        while(!ifs.eof())
        {
            string lines,segs;
            getline(ifs,lines);
            if(ifs.eof()) {break;}
            stringstream ss_lines(lines);
            vector<string> slices;
            while (ss_lines >> segs)
            {
                if(segs.size()!=0)
                {
                    slices.push_back(segs);
                }
            }
            int num_element = slices.size();
            if(num_element<2){
                break;
            }
            int num_ops = (num_element - 1) / 2;
            vector<char> cur_type(nsites,'I');
            for(int loc=0;loc<num_ops;++loc)
            {
                cur_type[stoul(slices[loc])] = slices[loc+num_ops][0];
            }
            rvals.push_back(stod(slices[2*num_ops]));
            types.push_back(cur_type);
            if(ifs.eof()) {break;}
        }
        ifs.close();
    }
    MPO<D> gen_local_cop(char op_type, int site)
    {
        MPO<D> res(nsites);
        for(int j=0;j<nsites;++j)
        {
            if(j==site)
            {
                res.tt[j] = to_matrix(op_type,nphysdims[j]);
            }
            else
            {
                res.tt[j] = to_matrix('I',nphysdims[j]);
            }
        }
        return res;
    }
    MPO<D> gen_global_op(vector<char> const & ops)
    {
        MPO<D> res(nsites);
	for(int j=0;j<nsites;++j)
	{
	    res.tt[j] = to_matrix(ops[j],nphysdims[j]);
	}
	return res;
    }
    MPO<D> gen_total_para()
    {
        int nTerm = rvals.size(), nth;
        #pragma omp parallel
        nth = omp_get_num_threads();

        vector<MPO<D>> hams(nth,nsites);
        #pragma omp parallel for
        for(int i=0;i<nTerm;++i)
        {
            MPO<D> tmpi(nsites);
            for(int j=0;j<nsites;++j)
            {
                tmpi.tt[j] = to_matrix(types[i][j],nphysdims[j]);
            }
            D val(rvals[i]);
            tmpi.tt[0] = tmpi.tt[0] * val;
            int ThreadID = omp_get_thread_num();
            if(hams[ThreadID].tt[0].shape[1]!=tmpi.tt[0].shape[1]) { hams[ThreadID] = tmpi; }
            else { hams[ThreadID] += tmpi; }
            hams[ThreadID].deparallel();
        }
        for(int i=1;i<nth;++i)
        {
            hams[0] += hams[i];
            hams[0].deparallel();
        }
        return hams[0];
    }
    MPO<Z> gen_diag_exponential(double t1)
    {
        MPO<Z> res(nsites);
        int nTerm = rvals.size();
        Z ones(0.0,-1.0);
        for(int i=0;i<nTerm;++i)
        {
            if(all_of(types[i].begin(),types[i].end(), [](char c){return c == 'I' || c == 'N';}))
            {
                int idx = distance(types[i].begin(), find(types[i].begin(),types[i].end(),'N'));
                if(idx >= 25)
                {
                    rvals[i] = 0.0;
                }
                Sparse<Z,4> ri({1,nphysdims[idx],nphysdims[idx],1});
                for(int j=0;j<nphysdims[idx];++j)
                {
                    ri.add({0,j,j,0},exp(ones*rvals[i]*(t1*j)));
                }
                res.tt[idx] = ri;
            }
        }
        return res;
    }
};
