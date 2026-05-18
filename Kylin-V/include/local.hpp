#pragma once
#include "sparse.hpp"
Sparse<D,4> to_matrix(char op, int dim)
{
    Sparse<D,4> res({1,dim,dim,1});
    int lev = 0;
    switch (op)
    {
    case 'I':
        for(lev=0;lev<dim;++lev)
        {
            res.add({0,lev,lev,0},1.0);
        }
        break;
    case 'N':
        for(lev=0;lev<dim;++lev)
        {
            res.add({0,lev,lev,0},1.0*lev);
        }
        break;
    case 'Q':
        for(lev=0;lev<dim-1;++lev)
        {
            res.add({0,lev,lev+1,0},sqrt(0.5*lev+0.5));
            res.add({0,lev+1,lev,0},sqrt(0.5*lev+0.5));
        }
        break;
    case '+':
        for(lev=0;lev<dim-1;++lev)
        {
            res.add({0,lev+1,lev,0},1.0*sqrt(1.0*lev+1.0));
        }
        break;
    case '-':
        for(lev=0;lev<dim-1;++lev)
        {
            res.add({0,lev,lev+1,0},1.0*sqrt(1.0*lev+1.0));
        }
        break;
    case '^':
        for(lev=0;lev<dim-2;++lev)
        {
            res.add({0,lev,lev+2,0},sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
            res.add({0,lev+2,lev,0},sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
        }
        for(lev=0;lev<dim;++lev)
        {
            res.add({0,lev,lev,0},0.5*(2*lev+1.0));
        }
        break;
    default:
        break;
    }
    return res;
}
