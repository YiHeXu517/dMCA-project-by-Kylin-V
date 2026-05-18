#pragma once
#include "mpo.hpp"
struct TDVP
{
    MPS<Z> state_;
    MPO<D> ham_;
    vector<Sparse<D,6>> double_ham_;
    vector<Dense<Z,3>> envl_;
    vector<Dense<Z,3>> envr_;
    
    TDVP(MPS<Z> const & s0, MPO<D> const & op0):state_(s0),ham_(op0),double_ham_(op0.tt.size()-1)
    {
        Dense<Z,3> edge({1,1,1});
        edge.ptr[0] = 1.0;
        envl_.push_back(edge);
        envr_.push_back(edge);
        size_t ns = s0.tt.size();
        for(size_t s=ns-1;s>1;--s)
        {
            envr_.push_back(sweep(envr_.back(),state_.tt[s],ham_.tt[s],state_.tt[s],'l'));
        }
        for(size_t s=0;s<ns-1;++s)
        {
            double_ham_[s] = oo(ham_.tt[s],ham_.tt[s+1]);
        }
    }
    ~TDVP() = default;
    template<size_t N>
    Dense<Z,N> local_evolve(size_t site, Dense<Z,N> const & s0, double dt, double tol=1e-8, int nk=10)
    {
        vector<Dense<Z,N>> Vm = {s0};
        double beta = s0.norm();
        Vm[0] *= 1.0 / beta;
        double err = 1.0;
        vector<tuple<int,int,Z>> trips;
        for(int k=0;k<nk;++k)
        {
            for(int j=0;j<k;++j)
            {
                Z ov = Vm[j].overlap(Vm[k]);
                ov *= -1.0;
                Vm[k] += Vm[j] * ov;
            }
            Vm[k] *= 1.0 / Vm[k].norm();
            Dense<Z,N> hv;
            if constexpr(N==2)
            {
                hv = multiply(envl_.back(),envr_.back(),Vm[k]);
            }
            else if constexpr(N==3)
            {
                hv = multiply(envl_.back(),ham_.tt[site],envr_.back(),Vm[k]);
            }
            else
            {
                hv = multiply(envl_.back(),double_ham_[site],envr_.back(),Vm[k]);
            }
            hv *= dt;
            for(int j=max(0,k-1);j<k+1;++j)
            {
                Z ov = Vm[j].overlap(hv);
                trips.push_back(make_tuple(j,k,ov));
                ov *= -1.0;
                hv += Vm[j] * ov;
            }
            double nmhv = hv.norm();
            err *= nmhv;
            if(err<=tol || k==nk-1)
            {
                break;
            }
            else
            {
                trips.push_back(make_tuple(k+1,k,(Z)nmhv));
                hv *= 1.0 / nmhv;
                Vm.push_back(hv);
            }
        }
        int Nm = Vm.size();
        Dense<Z,2> Hm({Nm,Nm});
        for(auto const & tp : trips)
        {
            Hm({get<0>(tp),get<1>(tp)}) = get<2>(tp);
        }
        //if constexpr(N==2){ cout << "Site = " << site << endl; Hm.print();  }
        Dense<Z,2> eaHm = eig<Z>(Hm);
        Z ones(0,-1.0);
        for(int m=0;m<Nm;++m)
        {
            eaHm({m,m}) = exp(ones*eaHm({m,m}));
        }
        eaHm = mm<Z,2,2,1>(Hm,eaHm);
        eaHm = mm<Z,2,2,1>(eaHm,Hm,CblasNoTrans,CblasConjTrans);
        Dense<Z,N> res = Vm[0] * eaHm({0,0});
        for(int m=1;m<Nm;++m)
        {
            res += Vm[m] * eaHm({m,0});
        }
        res *= beta;
        return res;
    }
    void impl_single(double dt, double tol=1e-8, int dims=1000)
    {
        size_t ns = ham_.tt.size();
        for(size_t site=0;site<ns;++site)
        {
            Dense<Z,3> s2 = local_evolve<3>(site,state_.tt[site],0.5*dt);
            if(site!=ns-1)
            {
                auto[lef,rig] = svd<Z,2,1>(s2,'r',tol);
                state_.tt[site] = move(lef);
                envl_.push_back(sweep(envl_.back(),state_.tt[site],ham_.tt[site],state_.tt[site],'r'));
                rig = local_evolve<2>(site+1,rig,-0.5*dt);
                Dense<Z,3> nex = transpose<Z,3>(state_.tt[site+1],{1,0,2});
                nex = mm<Z,2,3,1>(rig,nex);
                state_.tt[site+1] = transpose<Z,3>(nex,{1,0,2});
                envr_.pop_back();
            }
            else
            {
                state_.tt[site] = move(s2);
            }
        }
        for(size_t site=ns;site>0;--site)
        {
            Dense<Z,3> s2 = local_evolve<3>(site-1,state_.tt[site-1],0.5*dt);
            if(site!=1)
            {
                s2 = transpose<Z,3>(s2,{1,0,2});
                auto[lef,rig] = svd<Z,1,2>(s2,'l',tol);
                state_.tt[site-1] = transpose<Z,3>(rig,{1,0,2});
                envr_.push_back(sweep(envr_.back(),state_.tt[site-1],ham_.tt[site-1],state_.tt[site-1],'l'));
                lef = local_evolve<2>(site-1,lef,-0.5*dt);
                state_.tt[site-2] = mm<Z,3,2,1>(state_.tt[site-2],lef);
                envl_.pop_back();
            }
            else
            {
                state_.tt[site-1] = move(s2);
            }
        }
    }
    void impl_double(double dt, double tol=1e-8, int dims=1000)
    {
        size_t ns = ham_.tt.size();
        for(size_t site=0;site<ns-1;++site)
        {
            Dense<Z,4> s2 = ss(state_.tt[site],state_.tt[site+1]);
            s2 = local_evolve<4>(site,s2,0.5*dt);
            s2 = transpose<Z,4>(s2,{2,0,1,3});
            auto[lef,rig] = svd<Z,2,2>(s2,'r',tol,dims);
            if(site!=ns-2)
            {
                state_.tt[site] = transpose<Z,3>(lef,{1,0,2});
                envl_.push_back(sweep(envl_.back(),state_.tt[site],ham_.tt[site],state_.tt[site],'r'));
                rig = transpose<Z,3>(rig,{1,0,2});
                state_.tt[site+1] = local_evolve<3>(site+1,rig,-0.5*dt);
                envr_.pop_back();
            }
            else
            {
                state_.tt[site] = transpose<Z,3>(lef,{1,0,2});
                state_.tt[site+1] = transpose<Z,3>(rig,{1,0,2});
            }
        }
        for(size_t site=ns;site>1;--site)
        {
            Dense<Z,4> s2 = ss(state_.tt[site-2],state_.tt[site-1]);
            s2 = local_evolve<4>(site-2,s2,0.5*dt);
            s2 = transpose<Z,4>(s2,{2,0,1,3});
            auto[lef,rig] = svd<Z,2,2>(s2,'l',tol,dims);
            if(site!=2)
            {
                state_.tt[site-1] = transpose<Z,3>(rig,{1,0,2});
                envr_.push_back(sweep(envr_.back(),state_.tt[site-1],ham_.tt[site-1],state_.tt[site-1],'l'));
                lef = transpose<Z,3>(lef,{1,0,2});
                state_.tt[site-2] = local_evolve<3>(site-2,lef,-0.5*dt);
                envl_.pop_back();
            }
            else
            {
                state_.tt[site-1] = transpose<Z,3>(rig,{1,0,2});
                state_.tt[site-2] = transpose<Z,3>(lef,{1,0,2});
            }
        }
    }
};
