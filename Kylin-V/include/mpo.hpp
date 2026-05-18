#pragma once
#include "sparse.hpp"
#include "mps.hpp"
template<typename T>
struct MPO
{ 
    vector<Sparse<T,4>> tt;
    MPO() = default;
    MPO(MPO<T> const & r):tt(r.tt)
    {
    
    }
    MPO(MPO<T> && r):tt(r.tt)
    {
        
    }
    MPO(size_t ns):tt(ns)
    {
    
    }
    ~MPO() = default;
    MPO<T> & operator=(MPO<T> const & r)
    {
        tt = r.tt;
	return *this;
    }
    MPO<T> & operator=(MPO<T> && r)
    {
        tt = move(r.tt);
	return *this;
    }
    MPO<T> & operator+=(MPO<T> const & r)
    {
        size_t ns = tt.size();
        tt[0] = stack<T,4>(tt[0],r.tt[0],{3});
        tt[ns-1] = stack<T,4>(tt[ns-1],r.tt[ns-1],{0});
        for(size_t s=1;s<ns-1;++s)  
        {
            tt[s] = stack<T,4>(tt[s],r.tt[s],{0,3});
        }
        return *this;
    }
    MPO<T> operator+(MPO<T> const & r)
    {
        MPO<T> res(*this);
        res += r;
        return res;
    }
    MPO<T> & operator*=(double val)
    {
        tt[0] *= val;
        return *this;
    }
    MPO<T> operator*(T const & val)
    {
        MPO<T> res(*this);
        res.tt[0] = res.tt[0] * val;
        return res;
    }
    void print() const
    {  
        size_t ns = tt.size();
        cout << ns << "-site MPO" << endl;
        for(size_t s=0;s<ns;++s)
        {
            cout << "Site " << s+1 << endl;
            tt[s].print();
        }
    }
    void save(char const * file_name) const
    {
        ofstream ofs(file_name);
        size_t ns = tt.size();
        ofs.write(reinterpret_cast<char*>(&ns),sizeof(size_t));
        for(size_t s=0;s<ns;++s)
        {
            ofs.write(reinterpret_cast<char const*>(tt[s].shape.data()),4*sizeof(int));
            size_t npos = tt[s].pos.size();
            ofs.write(reinterpret_cast<char const*>(&npos),sizeof(size_t));
            for(size_t ip=0;ip<npos;++ip)
            {
                ofs.write(reinterpret_cast<char const*>(tt[s].pos[ip].data()),4*sizeof(int));
                ofs.write(reinterpret_cast<char const*>(&tt[s].data[ip]),sizeof(T));
            }
        }
        ofs.close();
    }
    static MPO<T> load(char const * file_name)
    {
        ifstream ifs(file_name);
        size_t ns;
        ifs.read(reinterpret_cast<char*>(&ns),sizeof(size_t));
        MPO<T> res(ns);
        for(size_t s=0;s<ns;++s)
        {
            array<int,4> sp;
            ifs.read(reinterpret_cast<char*>(sp.data()),4*sizeof(int));
            Sparse<T,4> ri(sp);
            size_t npos;
            ifs.read(reinterpret_cast<char*>(&npos),sizeof(size_t));
            for(size_t ip=0;ip<npos;++ip)
            {
                array<int,4> rpos; 
                T rdata;
                ifs.read(reinterpret_cast<char *>(rpos.data()),4*sizeof(int));
                ifs.read(reinterpret_cast<char *>(&rdata),sizeof(T));
                ri.add(rpos,rdata);
            }
            res.tt[s] = move(ri);
        }
        ifs.close();
        return res;
    }
    void deparallel(double tol=1e-14)
    {
        size_t ns = tt.size();
        for(size_t s=0;s<ns-1;++s)
        {
            auto[lef,rig] = decompose<T,4>(tt[s],'r',tol);
            tt[s] = move(lef);
            tt[s+1] = contract<T,4>(tt[s+1],rig,'r');
        }
        for(size_t s=ns-1;s>0;--s)
        {
            auto[lef,rig] = decompose<T,4>(tt[s],'l',tol);
            tt[s] = move(lef);
            tt[s-1] = contract<T,4>(tt[s-1],rig,'l');
        }
    }
};
Z expectation(MPS<Z> const & s1, MPO<D> const & op, MPS<Z> const & s2)
{
    Dense<Z,3> env({1,1,1});
    env.ptr[0] = 1.0;
    size_t ns = op.tt.size();
    for(size_t s=0;s<ns;++s)
    {
        env = sweep(env,s1.tt[s],op.tt[s],s2.tt[s],'r');
    }
    return env.ptr[0];
}
Z expectation(MPS<Z> const & s1, MPO<Z> const & op, MPS<Z> const & s2)
{
    Dense<Z,3> env({1,1,1});
    env.ptr[0] = 1.0;
    size_t ns = op.tt.size();
    for(size_t s=0;s<ns;++s)
    {
        env = sweep(env,s1.tt[s],op.tt[s],s2.tt[s],'r');
    }
    return env.ptr[0];
}
MPS<Z> multiply(MPO<D> const & op, MPS<Z> const & s0, double tol = 1e-14, int ndims = 1000)
{
    Dense<Z,3> env({1,1,1});
    env.ptr[0] = 1.0;
    size_t ns = op.tt.size();
    MPS<Z> res(ns);
    for(size_t s=0;s<ns;++s)
    {
        Dense<Z,4> lso = env_mps_mpo(env, op.tt[s], s0.tt[s]);
        lso = transpose<Z,4>(lso,{0,2,1,3});
        if(s!=ns-1)
        {
            auto[lef,rig] = svd<Z,2,2>(lso,'r',tol,ndims);
            res.tt[s] = move(lef);
            env = transpose<Z,3>(rig,{1,0,2});
        }
        else
        {
            Dense<Z,3> ss({lso.shape[0],lso.shape[1],1});
            memcpy(ss.ptr,lso.ptr,lso.size*sizeof(Z));
            res.tt[ns-1] = move(ss);
        }
    }
    for(size_t s=ns-1;s>0;--s)
    {
        Dense<Z,3> ttst = transpose<Z,3>(res.tt[s],{1,0,2});
        auto[lef,rig] = svd<Z,1,2>(ttst,'l',tol,1000);
        res.tt[s] = transpose<Z,3>(rig,{1,0,2});
        res.tt[s-1] = mm<Z,3,2,1>(res.tt[s-1],lef);
    }
    return res;
}
