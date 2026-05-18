#pragma once
#include "util.hpp"
template<typename T, int N> struct Dense
{
    array<int,N> shape;
    array<int,N> dist;
    int size = 0;
    T * ptr = nullptr;

    Dense() = default;
    ~Dense()
    {
        free(ptr);
        ptr = nullptr;
    }
    Dense(Dense<T,N> const & r)
    :shape(r.shape),dist(r.dist),size(r.size)
    {
        ptr = (T *)malloc(size*sizeof(T));
        memcpy(ptr,r.ptr,size*sizeof(T));
    }
    Dense(Dense<T,N> && r)
    :shape(move(r.shape)),dist(move(r.dist)),size(r.size)
    {
        ptr = r.ptr;
        r.ptr = nullptr;
    }
    Dense(array<int,N> const & other_shape)
    :shape(other_shape),size(accumulate(other_shape.begin(),other_shape.end(),1 ,multiplies<int>()))
    {
        dist[N-1] = 1;
        for(int ax=1;ax<N;++ax)
        {
            dist[N-1-ax] = dist[N-ax] * shape[N-ax];
        }
        ptr = (T *)calloc(size,sizeof(T));
    }
    Dense(initializer_list<int> other_shape)
    :size(accumulate(other_shape.begin(),other_shape.end(),1,multiplies<int>()))
    {
        copy(other_shape.begin(),other_shape.end(),shape.begin());
        dist[N-1] = 1;
        for(int ax=1;ax<N;++ax)
        {
            dist[N-1-ax] = dist[N-ax] * shape[N-ax];
        }
        ptr = (T *)calloc(size,sizeof(T));
    }
    Dense<T,N> & operator=(Dense<T,N> const & r)
    {
        free(ptr);
        shape = r.shape;
        dist = r.dist;
        size = r.size;
        ptr = (T *)malloc(size*sizeof(T));
        memcpy(ptr,r.ptr,size*sizeof(T));
        return *this;
    }
    Dense<T,N> & operator=(Dense<T,N> && r)
    {
        free(ptr);
        shape = move(r.shape);
        dist = move(r.dist);
        size = move(r.size);
        ptr = r.ptr;
        r.ptr = nullptr;           
        return *this;
    }
    T & operator()(array<int,N> const & idx)
    {
        int pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
        return ptr[pos];
    }
    T & operator()(initializer_list<int> idx)
    {
        int pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
        return ptr[pos];
    }
    T const & operator()(array<int,N> const & idx) const
    {
        int pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
        return ptr[pos];
    }
    T const & operator()(initializer_list<int> idx) const
    {
        int pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
        return ptr[pos];
    }
    Dense<T,N> & operator+=(Dense<T,N> const & r)
    {
        T ones(1.0);
        if constexpr(is_same<T,double>::value)
        {
            cblas_daxpy(size,ones,r.ptr,1,ptr,1);
        }
        else
        {
            cblas_zaxpy(size,&ones,r.ptr,1,ptr,1);
        }
        return *this;
    }
    Dense<T,N> operator+(Dense<T,N> const & r) const
    {
        Dense<T,N> res(*this);
        res += r;
        return res;
    }
    Dense<T,N> & operator-=(Dense<T,N> const & r)
    {
        T ones(-1.0);
        if constexpr(is_same<T,double>::value)
        {
            cblas_daxpy(size,ones,r.ptr,1,ptr,1);
        }
        else
        {
            cblas_zaxpy(size,&ones,r.ptr,1,ptr,1);
        }
        return *this;
    }
    Dense<T,N> operator-(Dense<T,N> const & r) const
    {
        Dense<T,N> res(*this);
        res -= r;
        return res;
    }
    Dense<T,N> & operator*=(double val)
    {
        if constexpr(is_same<T,double>::value)
        {
            cblas_dscal(size,val,ptr,1);
        }
        else
        {
            cblas_zdscal(size,val,ptr,1);
        }
        return *this;
    }
    Dense<T,N> operator*(T const & val) const
    {
        Dense<T,N> res(*this);
        if constexpr(is_same<T,double>::value)
        {
            cblas_dscal(size,val,res.ptr,1);
        }
        else
        {
            cblas_zscal(size,&val,res.ptr,1);
        }
        return res;
    }
    Dense<T,N> & operator/=(double val)
    {
        if constexpr(is_same<T,double>::value)
        {
            cblas_dscal(size,1/val,ptr,1);
        }
        else
        {
            cblas_zdscal(size,1/val,ptr,1);
        }
        return *this;
    }
    Dense<T,N> operator/(T const & val) const
    {
        Dense<T,N> res(*this);
        T valc(1.0/val);
        if constexpr(is_same<T,double>::value)
        {
            cblas_dscal(size,valc,res.ptr,1);
        }
        else
        {
            cblas_zscal(size,&valc,res.ptr,1);
        }
        return res;
    }
    void print(double tol=1e-15) const
    {
        cout << "Shape:";
        cout << shape << endl;  
        for(int idx=0;idx<size;++idx)
        {
            array<int,N> idices;
            if(abs(ptr[idx])<=tol)
            {
                continue;
            }
            for(int ax=0;ax<N;++ax)
            {
                idices[ax] = idx / dist[ax] % shape[ax];
            }
            cout << idices << " | " << std::scientific << ptr[idx] << endl;
        }
    }
    double norm() const
    {
        double res = 0.0;
        if constexpr(is_same<T,double>::value)
        {
            res = cblas_dnrm2(size,ptr,1);
        }
        else
        {
            res = cblas_dznrm2(size,ptr,1);
        }
        return res;
    }
    T overlap(Dense<T,N> const & r) const
    {
        T res(0.0);
        if constexpr(is_same<T,double>::value)
        {
            res = cblas_ddot(size,ptr,1,r.ptr,1);
        }
        else
        {
            cblas_zdotc_sub(size,ptr,1,r.ptr,1,&res);
        }
        return res;
    }
};
