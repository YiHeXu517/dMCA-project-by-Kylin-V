#pragma once
#include "dense.hpp"
template<typename T, int N>
Dense<T,N> transpose(Dense<T,N> const & r, array<int,N> const & ax)
{
    array<int,N> rsp;
    for(int i=0;i<N;++i)
    {
        rsp[i] = r.shape[ax[i]];
    } 
    Dense<T,N> res(rsp);
    for(int i=0;i<r.size;++i)
    {
        array<int,N> idx;
        for(int j=0;j<N;++j)
        {
            idx[j] = i / r.dist[ax[j]] % r.shape[ax[j]];
        }
        res(idx) = r.ptr[i];
    }
    return res;
}
template<typename T, int N>
Dense<T,N> transpose(Dense<T,N> const & r, initializer_list<int> ax)
{
    array<int,N> axr;
    copy(ax.begin(),ax.end(),axr.begin());
    return transpose<T,N>(r,axr);
}
// stack
template<typename T, int N>
Dense<T,N> stack(Dense<T,N> const & r1, Dense<T,N> const & r2, initializer_list<int> ax)
{
    array<int,N> rsp(r1.shape);
    for(auto it = ax.begin(); it!= ax.end(); ++it)
    {
        rsp[*it] += r2.shape[*it];
    }
    Dense<T,N> res(rsp);
    for(int i=0;i<r1.size;++i)
    {
        array<int,N> idx;
        for(int j=0;j<N;++j)
        {
            idx[j] = i / r1.dist[j] % r1.shape[j];
        }
        res(idx) = r1.ptr[i];
    }
    for(int i=0;i<r2.size;++i)
    {
        array<int,N> idx;
        for(int j=0;j<N;++j)
        {
            idx[j] = i / r2.dist[j] % r2.shape[j];
        }
        for(auto it = ax.begin(); it!= ax.end(); ++it)
        {
            idx[*it] += r1.shape[*it];
        }
        res(idx) = r2.ptr[i];
    }
    return res;
}
template<typename T, int R1, int R2, int rc> Dense<T,R1+R2-2*rc> mm(Dense<T,R1> const & m1, Dense<T,R2> const & m2, 
CBLAS_TRANSPOSE t1 = CblasNoTrans, CBLAS_TRANSPOSE t2 = CblasNoTrans) 
{
    array<int,R1+R2-2*rc> rsp;
    int nr, nc, nk;
    int lda, ldb;
    if(t1==CblasNoTrans)
        {
        copy(m1.shape.begin(),m1.shape.begin()+R1-rc,rsp.begin());
        nr = accumulate(m1.shape.begin(),m1.shape.begin()+R1-rc,1,multiplies<int>());
        lda = m1.size / nr;
    }
    else
    {
        copy(m1.shape.begin()+rc,m1.shape.end(),rsp.begin());
        nr = accumulate(m1.shape.begin()+rc,m1.shape.end(),1,multiplies<int>());
        lda = nr;
    }
    if(t2==CblasNoTrans)
    {
        copy(m2.shape.begin()+rc,m2.shape.end(),rsp.begin()+R1-rc);
        nc = accumulate(m2.shape.begin()+rc,m2.shape.end(),1,multiplies<int>());
        ldb = nc;
    }
    else
    {
        copy(m2.shape.begin(),m2.shape.begin()+R2-rc,rsp.begin()+R1-rc);
        nc = accumulate(m2.shape.begin(),m2.shape.begin()+R2-rc,1,multiplies<int>());
        ldb = m2.size / nc;
    }
    nk = m1.size / nr;
    if(nk!=m2.size / nc)
    {
        cout << "shapes mismatch!" << endl;
        m1.print(1e9);
        m2.print(1e9);
        cout << rc << endl;
        exit(1);
    }
    T ones(1.0),zeros(0.0);
    Dense<T,R1+R2-2*rc> res(rsp);
    if constexpr(is_same<T,double>::value)
    {
        cblas_dgemm(CblasRowMajor, t1, t2, nr, nc, nk, ones,
            m1.ptr, lda, m2.ptr, ldb, zeros, res.ptr, nc);
    }
    else
    {
        cblas_zgemm(CblasRowMajor, t1, t2, nr, nc, nk, &ones,
            m1.ptr, lda, m2.ptr, ldb, &zeros, res.ptr, nc);
    }
    return res;
}
template<typename T, int R1, int R2>
tuple<Dense<T,R1+1>,Dense<T,R2+1>> svd(Dense<T,R1+R2> & m, char l2r = 'r', double tol = 1e-14, int maxdim = 1000,
char ReS = 'n')
{
    int nrow = accumulate(m.shape.begin(),m.shape.begin()+R1,1,multiplies<int>());
    int ncol = m.size / nrow;
    int ldu = min(nrow,ncol);
    Dense<T,R1+R2> mc(m);
    Dense<T,2> u({nrow,ldu}),vt({ldu,ncol});
    Dense<double,1> s({ldu}),sp({ldu-1});
    array<int,R1+1> lsp; copy(m.shape.begin(),m.shape.begin()+R1,lsp.begin());
    array<int,R2+1> rsp; copy(m.shape.begin()+R1,m.shape.end(),rsp.begin()+1);
    if constexpr (is_same<T,double>::value)
    {
        int ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr, ncol, s.ptr, u.ptr, ldu, vt.ptr, ncol, sp.ptr);
        if(ifsv!=0) { cout << "Problems in SVD! " << R1 << " : " << R2 << endl; exit(1);}
    }
    else if constexpr (is_same<T,MKL_Complex16>::value)
    {
        int ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr, ncol, s.ptr, u.ptr, ldu, vt.ptr, ncol, sp.ptr);
        if(ifsv!=0) { cout << "Problems in SVD! " << R1 << " : " << R2 << endl; exit(1);}
    }
    int nstate = 1; double nrms = 0;
    for(int i=0;i<ldu;++i)
    {
        nrms += s.ptr[i];
        if(s.ptr[i]<tol)
        {
            break;
        }
        nstate = i + 1;
    }
    nstate = min(maxdim,nstate);
    lsp[R1] = nstate; rsp[0] = nstate;
    Dense<T,R1+1> lef(lsp); Dense<T,R2+1> rig(rsp);
    if constexpr (is_same<T,double>::value)
    {
        if(l2r=='r')
        {
            for(int i=0;i<nstate;++i)
            {
                cblas_dcopy(nrow, u.ptr+i, ldu, lef.ptr+i, nstate);
                cblas_daxpy(ncol, s.ptr[i], vt.ptr+i*ncol, 1, rig.ptr+i*ncol, 1);
            }
            if(ReS=='r') { lef *= sqrt(nrms); rig *= (1/sqrt(nrms));}
        }
        else
        {
            for(int i=0;i<nstate;++i)
            {
                cblas_daxpy(nrow, s.ptr[i],    u.ptr+i, ldu, lef.ptr+i, nstate);
                cblas_dcopy(ncol, vt.ptr+i*ncol, 1, rig.ptr+i*ncol, 1);
            }
            if(ReS=='r') { lef *= (1/sqrt(nrms)); rig *= sqrt(nrms); }
        }
    }
    else if constexpr (is_same<T,MKL_Complex16>::value)
    {
        if(l2r=='r')
        {
            for(int i=0;i<nstate;++i)
            {
                T sc(s.ptr[i]);
                cblas_zcopy(nrow, u.ptr+i, ldu, lef.ptr+i, nstate);
                cblas_zaxpy(ncol, &sc, vt.ptr+i*ncol, 1, rig.ptr+i*ncol, 1);
            }
            if(ReS=='r') { lef *= sqrt(nrms); rig *= (1/sqrt(nrms));}

        }
        else
        {
            for(int i=0;i<nstate;++i)
            {
                T sc(s.ptr[i]);
                cblas_zaxpy(nrow, &sc, u.ptr+i, ldu, lef.ptr+i, nstate);
                cblas_zcopy(ncol, vt.ptr+i*ncol, 1, rig.ptr+i*ncol, 1);
            }
            if(ReS=='r') { lef *= (1/sqrt(nrms)); rig *= sqrt(nrms); }
        }
    }
    return make_tuple(lef,rig);
}
// eigensolver
template<typename T>
Dense<T,2> eig(Dense<T,2> & Hm, char rep = 'r')
{
    int Ns = Hm.shape[0];
    int ifeg;
    Dense<double,1> ega({Ns});
    Dense<T,2> egr({Ns,Ns});
    if constexpr (is_same<T,double>::value)
    {
        ifeg = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr, Ns, ega.ptr);
        if(ifeg!=0) { cout << "Problems in eig! " << Ns << endl; exit(1);}
    }
    else if constexpr (is_same<T,MKL_Complex16>::value)
    {
        ifeg = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr, Ns, ega.ptr);
        if(ifeg!=0) { cout << "Problems in eig! " << Ns << endl; exit(1);}
    }
    for(int i=0;i<Ns;++i)
    {
        egr({i,i}) = ega({i});
    }
    return egr;                                                                                                                                                                                                                                                                                                                 
}