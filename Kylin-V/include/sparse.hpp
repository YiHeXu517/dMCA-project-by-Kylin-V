#pragma once
#include "util.hpp"
template<typename T,size_t N>
struct Sparse
{
    array<int,N> shape;
    vector<array<int,N>> pos;
    unordered_map<array<int,N>,size_t,array_hash<N>> ref;
    vector<T> data;
    Sparse() = default;
    Sparse(Sparse<T,N> const & r):shape(r.shape),pos(r.pos),ref(r.ref),data(r.data)
    {
    
    }
    Sparse(Sparse<T,N> && r):shape(move(r.shape)),pos(move(r.pos)),ref(move(r.ref)),data(move(r.data))
    {
    
    }
    Sparse(array<int,N> const & sp):shape(sp)
    {
        
    }
    Sparse(initializer_list<int> sp, vector<array<int,N>> const & ops, vector<T> const & oda)
    : pos(ops),data(oda)
    {
        copy(sp.begin(),sp.end(),shape.begin());
	for(size_t i=0;i<pos.size();++i)
	{
	    ref[pos[i]] = i;
	}
    }
    ~Sparse() = default;
    Sparse<T,N> & operator=(Sparse<T,N> const & r)
    {
        shape = r.shape;
        pos = r.pos;
        ref = r.ref;
        data = r.data;
        return *this;
    }
    Sparse<T,N> & operator=(Sparse<T,N> && r)
    {
        shape = move(r.shape);
        pos = move(r.pos);
        ref = move(r.ref);
        data = move(r.data);
        return *this;
    }
    Sparse<T,N> & operator+=(Sparse<T,N> const & r)
    {
        size_t nnz = r.pos.size();
        for(size_t i=0;i<nnz;++i)
        {
            add(r.pos[i],r.val[i]);
        }
        return *this;
    }
    Sparse<T,N> operator+(Sparse<T,N> const & r)
    {
        Sparse<T,N> res(*this);
        res += r;
        return res;
    }
    Sparse<T,N> & operator*=(D r)
    {
        size_t nnz = pos.size();
        for(size_t i=0;i<nnz;++i)
        {
            data[i] *= r;
        }
        return *this;
    }
    Sparse<T,N> operator*(T const & r)
    {
        Sparse<T,N> res(*this);
        size_t nnz = pos.size();
        for(size_t i=0;i<nnz;++i)
        {
            res.data[i] = res.data[i] * r;
        }
        return res;
    }
    void print() const
    {
        cout << "Shape:[ ";
        for(size_t ax=0;ax<N;++ax)
        {
           cout << shape[ax] << " ";
        }
        cout << "]" << endl;
        size_t nnz = pos.size();
        cout << nnz << " non-zero elements" << endl;
        for(size_t i=0;i<nnz;++i)
        {
            for(size_t ax=0;ax<N;++ax)   
            {
                cout << pos[i][ax] << ",";
            }
            cout << std::scientific << setprecision(8) << data[i] << endl;
        }
    }
    void add(array<int,N> const & idx, T const & val)
    {
        if(ref.find(idx)==ref.end())
        {
            ref[idx] = pos.size();
            pos.push_back(idx);
            data.push_back(val);
        }
        else
        {
            data[ref[idx]] += val;
        }
    }
    void add(initializer_list<int> idx, T const & val)
    {
        array<int,N> idxr;
        copy(idx.begin(),idx.end(),idxr.begin());
        add(idxr,val);
    }
    double norm() const
    {
        size_t nnz = pos.size();
        double res = 0.0;
        for(size_t i=0;i<nnz;++i)
        {
            res += pow(data[i], 2.0);
        }
        return sqrt(res);
    }
    T overlap(Sparse<T,N> const & r){
        size_t nnz = r.pos.size();
        T res(0.0);
        for(size_t i=0;i<nnz;++i)
        {
            if(ref.find(r.pos[i])!=ref.end())
            {
                res += data[ref[r.pos[i]]] * r.data[i];
            }
        }
        return res;
    }
};
template<typename T, size_t N>
Sparse<T,N> stack(Sparse<T,N> const & r1, Sparse<T,N> const & r2, initializer_list<size_t> axis)
{
    Sparse<T,N> res(r1);
    for(auto it=axis.begin();it!=axis.end();++it)
    {
        res.shape[*it] += r2.shape[*it];
    }
    for(size_t i=0;i<r2.pos.size();++i)
    {
        array<int,N> posr = r2.pos[i];
        for(auto it=axis.begin();it!=axis.end();++it)
        {
            posr[*it] += r1.shape[*it];
        }
        res.add(posr,r2.data[i]);
    }
    return res;
}
template<typename T, size_t N>
vector<Sparse<T,N-1>> slice(Sparse<T,N> const & r, char l2r, double tol=1e-14)
{
    if(l2r=='r')
    {
        size_t ncol = r.shape[N-1], nnz = r.pos.size();
        array<int,N-1> rsp;
        copy(r.shape.begin(),r.shape.begin()+N-1,rsp.begin());
        vector<Sparse<T,N-1>> res(ncol,rsp);
        for(size_t i=0;i<nnz;++i)
        {
            array<int,N-1> rpos;
            copy(r.pos[i].begin(),r.pos[i].begin()+N-1,rpos.begin());
            res[r.pos[i][N-1]].add(rpos,r.data[i]);
        }
        return res;
    }
    else
    {
        size_t nrow = r.shape[0], nnz = r.pos.size();
        array<int,N-1> rsp;
        copy(r.shape.begin()+1,r.shape.end(),rsp.begin());
        vector<Sparse<T,N-1>> res(nrow,rsp);
        for(size_t i=0;i<nnz;++i)
        {
            array<int,N-1> rpos;
            copy(r.pos[i].begin()+1,r.pos[i].end(),rpos.begin());
            res[r.pos[i][0]].add(rpos,r.data[i]);
        }
        return res;
    }
}
template<typename T, size_t N>
tuple<Sparse<T,N>,Sparse<T,2>> decompose(Sparse<T,N> const & r, char l2r, double tol=1e-14)
{
    vector<Sparse<T,N-1>> rcs = slice<T,N>(r,l2r,tol), rcs_truncated;
    if(l2r=='r')
    {
        size_t ncol = rcs.size();
        vector<array<int,2>> bond_pos;
        vector<T> bond_data;
        for(size_t col=0;col<ncol;++col)
        {
            char fd = 'n';
            double current_nm = rcs[col].norm();
            for(size_t col2=0;col2<rcs_truncated.size();++col2)
            {
                double current_nm2 = rcs_truncated[col2].norm();
                T ovp = rcs[col].overlap(rcs_truncated[col2]);
                if(abs(1.0-abs(ovp)/current_nm/current_nm2)<=tol || current_nm<=tol)
                {
                    fd = 'y';
                    array<int,2> bpos = {col2, col};
                    bond_pos.push_back(bpos);
                    bond_data.push_back(ovp/pow(current_nm2,2.0));
                    break;
                }
            }
            if(fd=='n')
            {
                rcs_truncated.push_back(rcs[col]);
                array<int,2> bpos = {rcs_truncated.size()-1,col};
                bond_pos.push_back(bpos);
                bond_data.push_back((T)1.0);
            }
        }
        int new_ncol = rcs_truncated.size();
        array<int,N> rsp(r.shape);
        rsp[N-1] = new_ncol;
        Sparse<T,N> res(rsp);
        Sparse<T,2> bond({new_ncol,ncol},bond_pos,bond_data);
        for(int nc=0;nc<new_ncol;++nc)
        {
            for(size_t i=0;i<rcs_truncated[nc].pos.size();++i)
            {
                array<int,N> rpos;
                rpos[N-1] = nc;
                copy(rcs_truncated[nc].pos[i].begin(),rcs_truncated[nc].pos[i].end(),rpos.begin());
                res.add(rpos,rcs_truncated[nc].data[i]);
            }
        }
        return make_tuple(res,bond);
    }
    else
    {
        size_t nrow = rcs.size();
        vector<array<int,2>> bond_pos;
        vector<T> bond_data;
        for(size_t row=0;row<nrow;++row)
        {
            char fd = 'n';
            double current_nm = rcs[row].norm();
            for(size_t row2=0;row2<rcs_truncated.size();++row2)
            {
                double current_nm2 = rcs_truncated[row2].norm();
                T ovp = rcs[row].overlap(rcs_truncated[row2]);
                if(abs(1.0-abs(ovp)/current_nm/current_nm2)<=tol || current_nm<=tol)
                {
                    fd = 'y';
                    array<int,2> bpos = {row, row2};
                    bond_pos.push_back(bpos);
                    bond_data.push_back(ovp/pow(current_nm2,2.0));
                    break;
                }
            }
            if(fd=='n')
            {
                rcs_truncated.push_back(rcs[row]);
                array<int,2> bpos = {row,rcs_truncated.size()-1};
                bond_pos.push_back(bpos);
                bond_data.push_back((T)1.0);
            }
        }
        int new_nrow = rcs_truncated.size();
        array<int,N> rsp(r.shape);
        rsp[0] = new_nrow;
        Sparse<T,N> res(rsp);
        Sparse<T,2> bond({nrow,new_nrow},bond_pos,bond_data);
        for(int nr=0;nr<new_nrow;++nr)
        {
            for(size_t i=0;i<rcs_truncated[nr].pos.size();++i)
            {
                array<int,N> rpos;
                rpos[0] = nr;
                copy(rcs_truncated[nr].pos[i].begin(),rcs_truncated[nr].pos[i].end(),rpos.begin()+1);
                res.add(rpos,rcs_truncated[nr].data[i]);
            }
        }
        return make_tuple(res,bond);
    }
}
template<typename T, size_t N>
Sparse<T,N> contract(Sparse<T,N> const & nex, Sparse<T,2> const & bond, char l2r)
{
    if(l2r=='r')
    {
        Sparse<T,N> res(nex.shape);
        res.shape[0] = bond.shape[0];
        for(size_t x=0;x<bond.pos.size();++x)
        {
            for(size_t y=0;y<nex.pos.size();++y)
            {
                if(nex.pos[y][0]!=bond.pos[x][1])
                {
                    continue;
                }
                array<int,N> rpos = nex.pos[y];
                rpos[0] = bond.pos[x][0];
                res.add(rpos,nex.data[y]*bond.data[x]);
            }
        }
        return res;
    }
    else
    {
        Sparse<T,N> res(nex.shape);
        res.shape[N-1] = bond.shape[1];
        for(size_t x=0;x<bond.pos.size();++x)
        {
            for(size_t y=0;y<nex.pos.size();++y)
            {
                if(nex.pos[y][N-1]!=bond.pos[x][0])
                {
                    continue;
                }
                array<int,N> rpos = nex.pos[y];
                rpos[N-1] = bond.pos[x][1];
                res.add(rpos,nex.data[y]*bond.data[x]);
            }
        }
        return res;
    }
}
