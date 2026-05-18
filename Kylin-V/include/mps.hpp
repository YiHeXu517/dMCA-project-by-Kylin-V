#pragma once
#include "product.hpp"
template<typename T> 
struct MPS
{
    vector<Dense<T,3>> tt;
    MPS() = default;
    MPS(MPS<T> const & r):tt(r.tt)
    {
    
    }
    MPS(MPS<T> && r):tt(r.tt)
    {
        
    }
    MPS(size_t ns):tt(ns)
    {
    
    }
    ~MPS() = default;
    MPS<T> & operator=(MPS<T> const & r)
    {
        tt = r.tt;
        return *this;
    }
    MPS<T> & operator=(MPS<T> && r)
    {
        tt = move(r.tt);
        return *this;
    }
    MPS<T> & operator+=(MPS<T> const & r)
    {
        size_t ns = tt.size();
        tt[0] = stack<T,3>(tt[0],r.tt[0],{2});
        tt[ns-1] = stack<T,3>(tt[ns-1],r.tt[ns-1],{1});
        for(size_t s=1;s<ns-1;++s)  
        {
            tt[s] = stack<T,3>(tt[s],r.tt[s],{1,2});
        }
        return *this;
    }
    MPS<T> operator+(MPS<T> const & r)
    {
        MPS<T> res(*this);
        res += r;
        return res;
    }
    MPS<T> & operator*=(double val)
    {
        tt[0] *= val;
        return *this;
    }
    MPS<T> operator*(T const & val)
    {
        MPS<T> res(*this);
        res[0] = res[0] * val;
        return *this;
    }
    void print() const
    {  
        size_t ns = tt.size();
        cout << ns << "-site MPS" << endl;
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
            ofs.write(reinterpret_cast<char const*>(tt[s].shape.data()),3*sizeof(int));
            ofs.write(reinterpret_cast<char const*>(tt[s].ptr),tt[s].size*sizeof(T));
        }
        ofs.close();
    }
    static MPS<T> load(char const * file_name)
    {
        ifstream ifs(file_name);
        size_t ns;
        ifs.read(reinterpret_cast<char*>(&ns),sizeof(size_t));
        MPS<T> res(ns);
        for(size_t s=0;s<ns;++s)
        {
            array<int,3> sp;
            ifs.read(reinterpret_cast<char*>(sp.data()),3*sizeof(int));
            Dense<T,3> ri(sp);
            ifs.read(reinterpret_cast<char*>(ri.ptr),ri.size*sizeof(T));
            res.tt[s] = move(ri);
        }
        ifs.close();
        return res;
    }
    void canon(double tol=1e-14)
    {
        size_t ns = tt.size();
        for(size_t s=0;s<ns-1;++s)
        {
            auto[lef,rig] = svd<T,2,1>(tt[s],'r',tol,1000);
            tt[s] = move(lef);
            tt[s+1] = transpose<T,3>(mm<T,2,3,1>(rig,transpose<T,3>(tt[s+1],{1,0,2})),{1,0,2});
        }
        for(size_t s=ns-1;s>0;--s)
        {
            Dense<T,3> ttst = transpose<T,3>(tt[s],{1,0,2});
            auto[lef,rig] = svd<T,1,2>(ttst,'l',tol);
            tt[s] = transpose<T,3>(rig,{1,0,2});
            tt[s-1] = mm<T,3,2,1>(tt[s-1],lef);
        }
    }
    tuple<size_t,int> max_bond() const
    {
        size_t site = 0;
        int b = 1;
        for(size_t s=0;s<tt.size();++s)
        {
            if(tt[s].shape[2]>b)
            {
                site = s;
                b = tt[s].shape[2];
            }
        }
        return make_tuple(site,b);
    }
};
Z overlap(MPS<Z> const & s1, MPS<Z> const & s2)
{
    Dense<Z,2> env({1,1});
    env.ptr[0] = 1.0;
    size_t ns = s1.tt.size();
    for(size_t s=0;s<ns;++s)
    {
        env = sweep(env,s1.tt[s],s2.tt[s],'r');
    }
    return env.ptr[0];
}
