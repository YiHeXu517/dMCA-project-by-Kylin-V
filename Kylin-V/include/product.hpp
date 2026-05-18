#pragma once
#include "linalg.hpp"
#include "sparse.hpp"
Dense<Z,2> sweep(Dense<Z,2> const & env, Dense<Z,3> const & s1, Dense<Z,3> const & s2, char l2r)
{
    if(l2r=='r')
    {
        Dense<Z,3> s1t = transpose<Z,3>(s1,{1,0,2});
        Dense<Z,3> s2t = transpose<Z,3>(s2,{1,0,2});
        Dense<Z,3> ls2 = mm<Z,2,3,1>(env,s2t);
        return mm<Z,3,3,2>(s1t,ls2,CblasConjTrans,CblasNoTrans);
    }
    else
    {
        Dense<Z,3> ls2 = mm<Z,2,3,1>(env,s2,CblasNoTrans,CblasTrans);
        Dense<Z,3> s1t = transpose<Z,3>(s1,{2,0,1});
        return mm<Z,3,3,2>(s1t,ls2,CblasConjTrans,CblasNoTrans);
    }
}
Dense<Z,4> env_mps_mpo(Dense<Z,3> const & env, Sparse<D,4> const & op, Dense<Z,3> const & s2)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,4>> ress(nth,{op.shape[1],op.shape[3],env.shape[1],s2.shape[2]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = env.shape[1], nc = s2.shape[2], nk = env.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        env.ptr+env.dist[0]*op.pos[i][0], nk, 
        s2.ptr+s2.dist[0]*op.pos[i][2], nc, 
        &zeros, m1.ptr, nc);

        cblas_zaxpy(m1.size,&val,m1.ptr,1,
        ress[thread_id].ptr+ress[thread_id].dist[0]*op.pos[i][1]+
        ress[thread_id].dist[1]*op.pos[i][3],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}
Dense<Z,4> env_mps_mpo(Dense<Z,3> const & env, Sparse<Z,4> const & op, Dense<Z,3> const & s2)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,4>> ress(nth,{op.shape[1],op.shape[3],env.shape[1],s2.shape[2]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = env.shape[1], nc = s2.shape[2], nk = env.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        env.ptr+env.dist[0]*op.pos[i][0], nk, 
        s2.ptr+s2.dist[0]*op.pos[i][2], nc, 
        &zeros, m1.ptr, nc);

        cblas_zaxpy(m1.size,&val,m1.ptr,1,
        ress[thread_id].ptr+ress[thread_id].dist[0]*op.pos[i][1]+
        ress[thread_id].dist[1]*op.pos[i][3],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}
Dense<Z,3> sweep(Dense<Z,3> const & env, Dense<Z,3> const & s1, Sparse<D,4> const & op, Dense<Z,3> const & s2, char l2r)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    if(l2r=='r')
    {
        vector<Dense<Z,3>> ress(nth,{op.shape[3],s1.shape[2],s2.shape[2]});
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<nnz_op;++i)
        {
            size_t thread_id = omp_get_thread_num();
            Z val(op.data[i]);

            int nr = env.shape[1], nc = s2.shape[2], nk = env.shape[2];
            Dense<Z,2> m1({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            env.ptr+env.dist[0]*op.pos[i][0], nk, 
            s2.ptr+s2.dist[0]*op.pos[i][2], nc, 
            &zeros, m1.ptr, nc);

            nr = s1.shape[2];
            nc = s2.shape[2]; 
            nk = s1.shape[1];
            Dense<Z,2> m2({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            s1.ptr+s1.dist[0]*op.pos[i][1], nr, 
            m1.ptr, nc, 
            &zeros, m2.ptr, nc);

            cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][3],1);
        }
        for(size_t i=1;i<nth;++i)
        {
            ress[0] += ress[i];
        }
        return ress[0];
    }
    else
    {
        Dense<Z,3> s1t = transpose<Z,3>(s1,{0,2,1});
        Dense<Z,3> s2t = transpose<Z,3>(s2,{0,2,1});
        vector<Dense<Z,3>> ress(nth,{op.shape[0],s1t.shape[2],s2t.shape[2]});
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<nnz_op;++i)
        {
            size_t thread_id = omp_get_thread_num();
            Z val(op.data[i]);

            int nr = env.shape[1], nc = s2t.shape[2], nk = env.shape[2];
            Dense<Z,2> m1({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            env.ptr+env.dist[0]*op.pos[i][3], nk, 
            s2t.ptr+s2t.dist[0]*op.pos[i][2], nc, 
            &zeros, m1.ptr, nc);

            nr = s1t.shape[2];
            nc = s2t.shape[2]; 
            nk = s1t.shape[1];
            Dense<Z,2> m2({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            s1t.ptr+s1t.dist[0]*op.pos[i][1], nr, 
            m1.ptr, nc, 
            &zeros, m2.ptr, nc);

            cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][0],1);
        }
        for(size_t i=1;i<nth;++i)
        {
            ress[0] += ress[i];
        }
        return ress[0];
    }
}
Dense<Z,3> sweep(Dense<Z,3> const & env, Dense<Z,3> const & s1, Sparse<Z,4> const & op, Dense<Z,3> const & s2, char l2r)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    if(l2r=='r')
    {
        vector<Dense<Z,3>> ress(nth,{op.shape[3],s1.shape[2],s2.shape[2]});
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<nnz_op;++i)
        {
            size_t thread_id = omp_get_thread_num();
            Z val(op.data[i]);

            int nr = env.shape[1], nc = s2.shape[2], nk = env.shape[2];
            Dense<Z,2> m1({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            env.ptr+env.dist[0]*op.pos[i][0], nk, 
            s2.ptr+s2.dist[0]*op.pos[i][2], nc, 
            &zeros, m1.ptr, nc);

            nr = s1.shape[2];
            nc = s2.shape[2]; 
            nk = s1.shape[1];
            Dense<Z,2> m2({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            s1.ptr+s1.dist[0]*op.pos[i][1], nr, 
            m1.ptr, nc, 
            &zeros, m2.ptr, nc);

            cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][3],1);
        }
        for(size_t i=1;i<nth;++i)
        {
            ress[0] += ress[i];
        }
        return ress[0];
    }
    else
    {
        Dense<Z,3> s1t = transpose<Z,3>(s1,{0,2,1});
        Dense<Z,3> s2t = transpose<Z,3>(s2,{0,2,1});
        vector<Dense<Z,3>> ress(nth,{op.shape[0],s1t.shape[2],s2t.shape[2]});
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<nnz_op;++i)
        {
            size_t thread_id = omp_get_thread_num();
            Z val(op.data[i]);

            int nr = env.shape[1], nc = s2t.shape[2], nk = env.shape[2];
            Dense<Z,2> m1({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            env.ptr+env.dist[0]*op.pos[i][3], nk, 
            s2t.ptr+s2t.dist[0]*op.pos[i][2], nc, 
            &zeros, m1.ptr, nc);

            nr = s1t.shape[2];
            nc = s2t.shape[2]; 
            nk = s1t.shape[1];
            Dense<Z,2> m2({nr,nc});
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            s1t.ptr+s1t.dist[0]*op.pos[i][1], nr, 
            m1.ptr, nc, 
            &zeros, m2.ptr, nc);

            cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][0],1);
        }
        for(size_t i=1;i<nth;++i)
        {
            ress[0] += ress[i];
        }
        return ress[0];
    }
}
Dense<Z,2> multiply(Dense<Z,3> const & envl, Dense<Z,3> const & envr, Dense<Z,2> const & s0)
{
    Dense<Z,3> lt = transpose<Z,3>(envl,{1,0,2}), rt = transpose<Z,3>(envr,{1,0,2});
    lt = mm<Z,3,2,1>(lt,s0);
    return mm<Z,3,3,2>(lt,rt,CblasNoTrans,CblasTrans);
}
Dense<Z,3> multiply(Dense<Z,3> const & envl, Sparse<D,4> const & op, Dense<Z,3> const & envr, Dense<Z,3> const & s0)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,3>> ress(nth,{op.shape[1],envl.shape[1],envr.shape[1]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = envl.shape[1], nc = s0.shape[2], nk = envl.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        envl.ptr+envl.dist[0]*op.pos[i][0], nk, 
        s0.ptr+s0.dist[0]*op.pos[i][2], nc, 
        &zeros, m1.ptr, nc);

        nr = envl.shape[1];
        nc = envr.shape[1]; 
        nk = envr.shape[2];
        Dense<Z,2> m2({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
        nr, nc, nk, &ones,
        m1.ptr, nk, 
        envr.ptr+envr.dist[0]*op.pos[i][3], nk, 
        &zeros, m2.ptr, nc);

        cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][1],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}
Dense<Z,3> multiply(Dense<Z,3> const & envl, Sparse<Z,4> const & op, Dense<Z,3> const & envr, Dense<Z,3> const & s0)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,3>> ress(nth,{op.shape[1],envl.shape[1],envr.shape[1]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = envl.shape[1], nc = s0.shape[2], nk = envl.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        envl.ptr+envl.dist[0]*op.pos[i][0], nk, 
        s0.ptr+s0.dist[0]*op.pos[i][2], nc, 
        &zeros, m1.ptr, nc);

        nr = envl.shape[1];
        nc = envr.shape[1]; 
        nk = envr.shape[2];
        Dense<Z,2> m2({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
        nr, nc, nk, &ones,
        m1.ptr, nk, 
        envr.ptr+envr.dist[0]*op.pos[i][3], nk, 
        &zeros, m2.ptr, nc);

        cblas_zaxpy(m2.size,&val,m2.ptr,1,ress[thread_id].ptr+m2.size*op.pos[i][1],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}
Sparse<D,6> oo(Sparse<D,4> const & op1, Sparse<D,4> const & op2)
{
    Sparse<D,6> res({op1.shape[0],op1.shape[1],op1.shape[2],op2.shape[1],op2.shape[2],op2.shape[3]});
    for(size_t x1=0;x1<op1.pos.size();++x1)
    {
        for(size_t x2=0;x2<op2.pos.size();++x2)
        {
            if( op1.pos[x1][3] == op2.pos[x2][0] )
            {
                res.add({op1.pos[x1][0],op1.pos[x1][1],op1.pos[x1][2],
                op2.pos[x2][1],op2.pos[x2][2],op2.pos[x2][3]},
                op1.data[x1]*op2.data[x2]);
            }
        }
    }
    return res;
}
Dense<Z,4> ss(Dense<Z,3> const & s1, Dense<Z,3> const & s2)
{
    Dense<Z,4> res({s1.shape[0],s2.shape[0],s1.shape[1],s2.shape[2]});
    Z ones(1.0),zeros(0.0);
    int nr = s1.shape[1], nc = s2.shape[2], nk = s1.shape[2];
    for(size_t x1=0;x1<s1.shape[0];++x1)
    {
        for(size_t x2=0;x2<s2.shape[0];++x2)
        {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nr, nc, nk, &ones,
            s1.ptr+x1*s1.dist[0], nk, 
            s2.ptr+x2*s2.dist[0], nc, 
            &zeros, res.ptr+x1*res.dist[0]+x2*res.dist[1], nc);
        }
    }
    return res;
}
Dense<Z,4> multiply(Dense<Z,3> const & envl, Sparse<D,6> const & op, Dense<Z,3> const & envr, Dense<Z,4> const & s0)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,4>> ress(nth,{op.shape[1],op.shape[3],envl.shape[1],envr.shape[1]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = envl.shape[1], nc = s0.shape[3], nk = envl.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        envl.ptr+envl.dist[0]*op.pos[i][0], nk, 
        s0.ptr+s0.dist[0]*op.pos[i][2]+s0.dist[1]*op.pos[i][4], nc,
        &zeros, m1.ptr, nc);

        nr = envl.shape[1];
        nc = envr.shape[1]; 
        nk = envr.shape[2];
        Dense<Z,2> m2({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
        nr, nc, nk, &ones,
        m1.ptr, nk, 
        envr.ptr+envr.dist[0]*op.pos[i][5], nk, 
        &zeros, m2.ptr, nc);

        cblas_zaxpy(m2.size,&val,m2.ptr,1,
        ress[thread_id].ptr+ress[thread_id].dist[0]*op.pos[i][1]+
        ress[thread_id].dist[1]*op.pos[i][3],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}

Dense<Z,4> multiply(Dense<Z,3> const & envl, Sparse<Z,6> const & op, Dense<Z,3> const & envr, Dense<Z,4> const & s0)
{
    size_t nnz_op = op.pos.size(), nth;
    Z ones(1.0),zeros(0.0);
    #pragma omp parallel
    nth = omp_get_num_threads();

    vector<Dense<Z,4>> ress(nth,{op.shape[1],op.shape[3],envl.shape[1],envr.shape[1]});
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<nnz_op;++i)
    {
        size_t thread_id = omp_get_thread_num();
        Z val(op.data[i]);

        int nr = envl.shape[1], nc = s0.shape[3], nk = envl.shape[2];
        Dense<Z,2> m1({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nr, nc, nk, &ones,
        envl.ptr+envl.dist[0]*op.pos[i][0], nk, 
        s0.ptr+s0.dist[0]*op.pos[i][2]+s0.dist[1]*op.pos[i][4], nc,
        &zeros, m1.ptr, nc);

        nr = envl.shape[1];
        nc = envr.shape[1]; 
        nk = envr.shape[2];
        Dense<Z,2> m2({nr,nc});
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
        nr, nc, nk, &ones,
        m1.ptr, nk, 
        envr.ptr+envr.dist[0]*op.pos[i][5], nk, 
        &zeros, m2.ptr, nc);

        cblas_zaxpy(m2.size,&val,m2.ptr,1,
        ress[thread_id].ptr+ress[thread_id].dist[0]*op.pos[i][1]+
        ress[thread_id].dist[1]*op.pos[i][3],1);
    }
    for(size_t i=1;i<nth;++i)
    {
        ress[0] += ress[i];
    }
    return ress[0];
}
