# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:35:11 2021

@author: wangh0m
"""
__author__ = 'Hui'
#------------------------------------------------------------------------------
import numpy as np

from scipy import sparse
#------------------------------------------------------------------------------
"""
from constraints_basic import 
    column3D,con_edge,con_unit,con_constl,con_equal_length,con_symmetry,
    con_planarity,con_planarity_constraints,con_unit_normal,
    con_orient,con_cross,con_osculating_tangent,
    con_ortho,con_orthogonal_2vectors,
    con_dependent_vector,con_equal_opposite_angle,con_constangle4,
    con_diagonal2,con_circle,
    con_constangle2,con_constangle3,con_positive,con_negative,con_orient1,con_orient2,
"""
#    # con_unique_angle1,con_unique_angle3,con_const_angle_cos1,con_const_angle_sin1,
    # con_multiply,con_unit_decomposition
# -------------------------------------------------------------------------
#                           general / basic
# -------------------------------------------------------------------------

def column3D(arr, num1, num2):
    """
    Parameters
    ----------
    array : array([1,4,7]).
    num1 : starting num.=100
    num2 : interval num.= 10

    Returns
    -------
    a : array(100+[1,4,7, 10,14,17, 20,24,27]).
    """
    a = num1 + np.r_[arr, num2+arr, 2*num2+arr]
    return a

def con_edge(X,c_v1,c_v3,c_ld1,c_ud1):
    "(v1-v3) = ld1*ud1"
    num = len(c_ld1)
    ld1 = X[c_ld1]
    ud1 = X[c_ud1]
    a3 = np.ones(3*num)
    row = np.tile(np.arange(3*num),4)
    col = np.r_[c_v1,c_v3,np.tile(c_ld1,3),c_ud1]
    data = np.r_[a3,-a3,-ud1,-np.tile(ld1,3)]
    r = -np.tile(ld1,3)*ud1
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, len(X)))
    return H,r

def con_unit_normal(X,c_e1,c_e2,c_e3,c_e4,c_n):
    "n^2=1; n*(e1-e3)=0; n*(e2-e4);"
    "Hui: better than (l*n=(e1-e3)x(e2-e4), but no orientation"
    H1,r1 = con_unit(X,c_n)
    H2,r2 = con_planarity(X,c_e1,c_e3,c_n)
    H3,r3 = con_planarity(X,c_e2,c_e4,c_n)
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    return H,r

def con_unit(X,c_ud1,w=100):
    "ud1**2=1"
    num = int(len(c_ud1)/3)
    arr = np.arange(num)
    row = np.tile(arr,3)
    col = c_ud1
    data = 2*X[col]
    r =  np.linalg.norm(X[col].reshape(-1,3,order='F'),axis=1)**2 + np.ones(num)
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H*w,r*w

def con_constl(c_ld1,init_l1,N):
    "ld1 == const."
    num = len(c_ld1)
    row = np.arange(num,dtype=int)
    col = c_ld1
    data = np.ones(num,dtype=int)
    r = init_l1
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_constangle2(X,c_u1,c_u2,c_a):
    "u1*u2 = a; a is 1 variable!"
    num = int(len(c_u1)/3)
    row = np.tile(np.arange(num),7)
    col = np.r_[c_u1,c_u2,c_a*np.ones(num)]
    data = np.r_[X[c_u2],X[c_u1],-np.ones(num)]
    r = np.einsum('ij,ij->i',X[c_u1].reshape(-1,3, order='F'),X[c_u2].reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_constangle3(X,c_on,c_vn,c_cos):
    "vN * oN2 - cos = 0; variables:vn,on2,cos; len(cos)=num"
    num = len(c_cos)
    row = np.tile(np.arange(num),7)
    data = np.r_[X[c_on],X[c_vn], -np.ones(num)]
    col = np.r_[c_vn,c_on,c_cos]
    r = np.einsum('ij,ij->i', X[c_vn].reshape(-1,3,order='F'), X[c_on].reshape(-1,3,order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_constangle4(X,c_on,n_xyz,sin):
    "on*n == const.; only on is variable; len(sin)=num=1/3len(n_xyz)=1/3len(c_on)"
    num = len(sin)
    row = np.tile(np.arange(num),3)
    col = c_on
    data = n_xyz
    r = sin
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_positive(X,c_K,c_a):
    "K=a^2"
    num = len(c_a)
    col = np.r_[c_a,c_K]
    row = np.tile(np.arange(num),2)
    data = np.r_[2*X[c_a],-np.ones(num)]
    r = X[c_a]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_negative(X,c_K,c_a,num):
    "K=-a^2"
    col = np.r_[c_a,c_K]
    row = np.tile(np.arange(num),2)
    data = np.r_[2*X[c_a],np.ones(num)]
    r = X[c_a]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_planarity(X,c_v1,c_v2,c_n): 
    "n*(v1-v2)=0"
    num = int(len(c_n)/3)
    col = np.r_[c_n,c_v1,c_v2]
    row = np.tile(np.arange(num),9)
    data = np.r_[X[c_v1]-X[c_v2],X[c_n],-X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),(X[c_v1]-X[c_v2]).reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_circle(X,c_v,c_c,c_r):
    "(v1-c)^2=r^2; (vertices, centers, radii are variables with same length)"
    num = len(c_r)
    row = np.tile(np.arange(num),7)
    col = np.r_[c_v,c_c,c_r]
    data = 2*np.r_[X[c_v]-X[c_c],X[c_c]-X[c_v],-X[c_r]]
    r = np.linalg.norm((X[c_v]-X[c_c]).reshape(-1,3,order='F'),axis=1)**2-X[c_r]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_equal_length(X,c1,c2,c3,c4):
    "(v1-v3)^2=(v2-v4)^2"
    num = int(len(c1)/3)
    row = np.tile(np.arange(num),12)
    col = np.r_[c1,c2,c3,c4]
    data = 2*np.r_[X[c1]-X[c3],X[c4]-X[c2],X[c3]-X[c1],X[c2]-X[c4]]
    r = np.linalg.norm((X[c1]-X[c3]).reshape(-1,3, order='F'),axis=1)**2
    r = r-np.linalg.norm((X[c2]-X[c4]).reshape(-1,3, order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X)))
    return H,r

def con_symmetry(X,cl,cc,cr):
    "(vl-vc)^2=(vr-vc)^2 <==> vl^2-vr^2-2*vl*vc+2*vr*vc=0"
    num = int(len(cc)/3)
    row = np.tile(np.arange(num),9)
    col = np.r_[cl,cc,cr]
    data = 2*np.r_[X[cl]-X[cc],X[cr]-X[cl],-X[cr]+X[cc]]
    r = np.linalg.norm((X[cl]-X[cc]).reshape(-1,3, order='F'),axis=1)**2
    r -= np.linalg.norm((X[cr]-X[cc]).reshape(-1,3, order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_equal_opposite_angle(X,c_e1,c_e2,c_e3,c_e4):
    "e1*e2-e3*e4=0"
    num = int(len(c_e1)/3)
    row = np.tile(np.arange(num),12)
    col = np.r_[c_e1,c_e2,c_e3,c_e4]
    data = np.r_[X[c_e2],X[c_e1], -X[c_e4], -X[c_e3]]
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X)))
    r = np.einsum('ij,ij->i',X[c_e1].reshape(-1,3, order='F'),X[c_e2].reshape(-1,3, order='F'))
    r -= np.einsum('ij,ij->i',X[c_e3].reshape(-1,3, order='F'),X[c_e4].reshape(-1,3, order='F'))
    return H,r

def con_orient(X,Nv,c_vN,c_a,neg=False):
    "vN*Nv = a^2; if neg: vN*Nv = -a^2; variables: vN, a; Nv is given"
    if neg:
        sign = -1
    else:
        sign = 1    
    num = int(len(c_a))
    row = np.tile(np.arange(num),4)
    col = np.r_[c_vN,c_a]
    data = np.r_[Nv.flatten('F'),-sign*2*X[c_a]]
    r = -sign*X[c_a]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
    return H,r

def con_orient1(X,c_on,c_vl,c_vr,c_b,neg=False):
    "on*(vl-vr)=b^2; if neg: on*(vl-vr)=-b^2"
    if neg:
        sign = -1
    else:
        sign = 1    
    num = int(len(c_b))
    row = np.tile(np.arange(num),10)
    col = np.r_[c_on,c_vl,c_vr,c_b]
    data = np.r_[X[c_vl]-X[c_vr],X[c_on],-X[c_on],-sign*2*X[c_b]]
    r = np.einsum('ij,ij->i',X[c_on].reshape(-1,3,order='F'),(X[c_vl]-X[c_vr]).reshape(-1,3,order='F'))
    r = r-sign*X[c_b]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
    return H,r

def con_orient2(X,c_on,n_xyz,c_b,neg=False):
    "on*n=b^2; if neg: on*n=-b^2; where on and b are variables; but n_xyz is given"
    if neg:
        sign = -1
    else:
        sign = 1    
    num = int(len(c_b))
    row = np.tile(np.arange(num),4)
    col = np.r_[c_on,c_b]
    data = np.r_[-sign*n_xyz,2*X[c_b]]
    r = X[c_b]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
    return H,r

def con_cross(X,c_a,c_b,c_n):
    "n = a x b = (a2b3-a3b2,a3b1-a1b3,a1b2-a2b3)"
    num = int(len(c_a)/3)
    arr = np.arange(num)
    one = np.ones(num)
    c_ax,c_ay,c_az = c_a[:num],c_a[num:2*num],c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num],c_b[num:2*num],c_b[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    
    row = np.tile(arr,5)
    def _cross(n1,a2,b3,a3,b2):
        "n1 = a2b3-a3b2"
        col = np.r_[a2,b3,a3,b2,n1]
        data = np.r_[X[b3],X[a2],-X[b2],-X[a3],-one]
        r = X[a2]*X[b3]-X[a3]*X[b2]
        H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
        return H,r
    H1,r1 = _cross(c_nx,c_ay,c_bz,c_az,c_by)
    H2,r2 = _cross(c_ny,c_az,c_bx,c_ax,c_bz)
    H3,r3 = _cross(c_nz,c_ax,c_by,c_ay,c_bx)
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    return H,r

def con_ortho(X,c_v1,c_v2,c_n):
    """
    n*(v1+v2)=0
    """
    num = int(len(c_n)/3)
    col = np.r_[c_n,c_v1,c_v2]
    row = np.tile(np.arange(num),9)
    data = np.r_[X[c_v1]+X[c_v2],X[c_n],X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),(X[c_v1]+X[c_v2]).reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_orthogonal_2vectors(X,c_n,c_e):
    "e*n = 0"
    num = int(len(c_n)/3)
    row = np.tile(np.arange(num),6)
    col = np.r_[c_n,c_e]
    data = np.r_[X[c_e],X[c_n]]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    r = np.einsum('ij,ij->i',X[c_e].reshape(-1,3, order='F'),X[c_n].reshape(-1,3, order='F'))
    return H,r

def con_osculating_tangent(X,c_v,c_v1,c_v3,c_ll1,c_ll3,c_lt,c_t,num):
    """ [ll1,ll3,lt,t]
        lt*t = l1**2*(V3-V0) - l3**2*(V1-V0)
        t^2=1
    <===>
        ll1 = l1**2 = (V1-V0)^2
        ll3 = l3**2 = (V3-V0)^2
        ll1 * (v3-v0) - ll3 * (v1-v0) - t*l = 0
        t^2=1
    """
    col = np.r_[c_v,c_v1,c_v3,np.tile(c_ll1,3),np.tile(c_ll3,3),c_t,np.tile(c_lt,3)]
    row = np.tile(np.arange(3*num),7)
    d_l1, d_l3 = X[np.tile(c_ll1,3)], X[np.tile(c_ll3,3)]
    d_v,d_v1,d_v3 = X[c_v], X[c_v1], X[c_v3]
    data = np.r_[-d_l1+d_l3, -d_l3, d_l1, d_v3-d_v, d_v-d_v1, -X[np.tile(c_lt,3)],-X[c_t]]
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num,len(X)))
    r = (d_v3-d_v)*X[np.tile(c_ll1,3)]-(d_v1-d_v)*X[np.tile(c_ll3,3)]
    r -= X[np.tile(c_lt,3)]*X[c_t]
    Hl1,rl1 = con_diagonal2(X,c_v,c_v1,c_ll1)
    Hl3,rl3 = con_diagonal2(X,c_v,c_v3,c_ll3)
    Hu,ru = con_unit(X, c_t)
    H = sparse.vstack((H,Hl1,Hl3,Hu))
    r = np.r_[r,rl1,rl3,ru]
    return H,r

def con_diagonal(X,c_v1,c_v3,c_d1):
    "(v1-v3)^2=d1^2"
    num = int(len(c_v1)/3)
    row = np.tile(np.arange(num),7)
    col = np.r_[c_v1,c_v3,c_d1*np.ones(num)]
    dd = X[c_d1]*np.ones(num,dtype=int)
    data = 2*np.r_[X[c_v1]-X[c_v3],X[c_v3]-X[c_v1],-dd]
    r = np.linalg.norm((X[c_v1]-X[c_v3]).reshape(-1,3,order='F'),axis=1)**2-dd**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_diagonal2(X,c_v1,c_v3,c_ll):
    "(v1-v3)^2=ll; len(c_ll)==1"
    num = int(len(c_v1)/3)
    row = np.tile(np.arange(num),7)
    col = np.r_[c_v1,c_v3,c_ll*np.ones(num)]
    data = 2*np.r_[X[c_v1]-X[c_v3],X[c_v3]-X[c_v1],-0.5*np.ones(num)]
    r = np.linalg.norm((X[c_v1]-X[c_v3]).reshape(-1,3,order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_dependent_vector(X,c_a,c_b,c_t):
    """ three variables: a, b,t
    t x (a-b) = 0
    <==> t1*(a-b)2=t2*(a-b)1; 
         t2*(a-b)3=t3*(a-b)2; 
         t1*(a-b)3=t3*(a-b)1
    """
    num = int(len(c_a)/3)
    arr = np.arange(num)
    c_ax,c_ay,c_az = c_a[:num], c_a[num:2*num], c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num], c_b[num:2*num], c_b[2*num:]
    c_tx,c_ty,c_tz = c_t[:num], c_t[num:2*num], c_t[2*num:]
    def _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty):
        "(ax-bx) * ty = (ay-by) * tx"
        col = np.r_[c_ax,c_ay,c_bx,c_by,c_tx,c_ty]
        row = np.tile(arr,6)
        data = np.r_[X[c_ty],-X[c_tx],-X[c_ty],X[c_tx],-X[c_ay]+X[c_by],X[c_ax]-X[c_bx]]
        r = (X[c_ax]-X[c_bx])*X[c_ty] - (X[c_ay]-X[c_by])*X[c_tx]
        H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
        return H,r
    H12,r12 = _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty)
    H23,r23 = _cross(c_az,c_ay,c_bz,c_by,c_tz,c_ty)
    H13,r13 = _cross(c_ax,c_az,c_bx,c_bz,c_tx,c_tz)
    H = sparse.vstack((H12,H23,H13))
    r = np.r_[r12,r23,r13]
    return H,r
    # -------------------------------------------------------------------------
    #                          Geometric Constraints (from Davide)
    # -------------------------------------------------------------------------

def con_normal_constraints(**kwargs):
    "represent unit normal: n^2=1"
    #w = kwargs.get('normal') * kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    F = mesh.F
    f = 3*V + np.arange(F)
    i = np.arange(F)
    i = np.hstack((i, i, i)) #row ==np.tile(i,3) == np.r_[i,i,i]
    j = np.hstack((f, F+f, 2*F+f)) #col ==np.r_[f,F+f,2*F+f]
    data = 2 * np.hstack((X[f], X[F+f], X[2*F+f])) #* w
    H = sparse.coo_matrix((data,(i,j)), shape=(F,len(X)))
    r = ((X[f]**2 + X[F+f]**2 + X[2*F+f]**2) + 1) #* w
    return H,r


def con_planarity_constraints(is_unit_edge=False,**kwargs):
    "n*(vi-vj) = 0; Note: making sure normals is always next to V in X[V,N]"
    w = kwargs.get('planarity')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    F = mesh.F
    f, v1, v2 = mesh.face_edge_vertices_iterators(order=True)
    if is_unit_edge:
        "f*(v1-v2)/length = 0, to avoid shrinkage of edges"
        num = len(v1)
        col_v1 = column3D(v1,0,V)
        col_v2 = column3D(v2,0,V)
        col_f = column3D(f,3*V,F)
        Ver = mesh.vertices
        edge_length = np.linalg.norm(Ver[v1]-Ver[v2],axis=1)
        row = np.tile(np.arange(num), 9)
        col = np.r_[col_f,col_v1,col_v2]
        l = np.tile(edge_length,3)
        data = np.r_[(X[col_v1]-X[col_v2])/l, X[col_f]/l, -X[col_f]/l]
        H = sparse.coo_matrix((data,(row, col)), shape=(num, len(X)))
        r = np.einsum('ij,ij->i',X[col_f].reshape(-1,3,order='F'),(X[col_v1]-X[col_v2]).reshape(-1,3,order='F'))
        r /= edge_length
        Hn,rn = con_unit(X,col_f,10*w)
        H = sparse.vstack((H*w,Hn))
        r = np.r_[r*w,rn]
    else:
        K = f.shape[0]
        f = 3*V + f
        r = ((X[v2] - X[v1]) * X[f] + (X[V+v2] - X[V+v1]) * X[F+f]
             + (X[2*V+v2] - X[2*V+v1]) * X[2*F+f] ) * w
        v1 = np.hstack((v1, V+v1, 2*V+v1))
        v2 = np.hstack((v2, V+v2, 2*V+v2))
        f = np.hstack((f, F+f, 2*F+f))
        i = np.arange(K)
        i = np.hstack((i, i, i, i, i, i, i, i, i))
        j = np.hstack((f, v2, v1))
        data = 2 * np.hstack((X[v2] - X[v1], X[f], -X[f])) * w
        H = sparse.coo_matrix((data,(i,j)), shape=(K, len(X)))
        Hn,rn = con_normal_constraints(**kwargs)
        H = sparse.vstack((H*w,Hn*w*10))
        r = np.r_[r*w,rn*w*10]
    return H,r
