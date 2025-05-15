# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:38:53 2022

@author: WANGH0M
"""
#---------------------------------------------------------------------------

import numpy as np

from geometrylab.geometry.meshpy import Mesh

#---------------------------------------------------------------------------

"""
make_quad_mesh_pieces
get_barycenters_mesh
get_strip_from_rulings
"""
#---------------------------------------------------------------------------
    
def make_quad_mesh_pieces(P1,P2,P3,P4):
    "fork from quadring.py"
    vlist = np.vstack((P1,P2,P3,P4))
    num = len(P1)
    arr = np.arange(num)
    flist = np.c_[arr,arr+num,arr+2*num,arr+3*num].tolist()
    ck = Mesh()
    ck.make_mesh(vlist,flist)
    return ck

def get_barycenters_mesh(mesh,verlist=None):
    "barycenteras vertices, new faces"
    H = mesh.halfedges
    if verlist is None:
        verlist = mesh.face_barycenters()
    v,vj,lj = mesh.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
    iv = np.where(lj==4)[0]
    flist = np.array([])
    for i in iv:
        ei = np.where(H[:,0]==i)[0]
        if -1 in H[ei,1]:
            continue
        e1 = np.where(H[:,0]==i)[0][0]
        e2 = H[H[e1,3],4]
        e3 = H[H[e2,3],4]
        e4 = H[H[e3,3],4]
        f1,f2,f3,f4 = H[e1,1],H[e2,1],H[e3,1],H[e4,1]
        flist = np.r_[flist,f1,f2,f3,f4]
    flist = flist.reshape(-1,4).tolist()   
    
    new = Mesh()
    new.make_mesh(verlist,flist)
    return new

def get_strip_from_rulings(an,ruling,row_list,is_smooth,is_even_selection=False,is_fix=False):
    "AG-NET: rectifying(tangent) planes along asymptotic(geodesic) crv."
    allV, allF = np.zeros(3), np.zeros(4,dtype=int)
    mmm = 0
    numv = 0

    #m = len(row_list)
    ck = 0
    for num in row_list:
        srr = mmm + np.arange(num)
        Pup = an[srr]+ruling[srr]
        if is_smooth:
            Pup = fair_vertices_or_vectors(Pup,itera=50,efair=is_smooth,is_fix=is_fix)
        P1234 = np.vstack((an[srr], Pup))
        arr1 = numv + np.arange(num)
        arr2 = arr1 + num
        flist = np.c_[arr1[:-1],arr1[1:],arr2[1:],arr2[:-1]]
        
        if is_even_selection:
            if ck %2 == 0:
            #if ck < m/2:
                allV, allF = np.vstack((allV,P1234)), np.vstack((allF,flist))
        else:
            allV, allF = np.vstack((allV,P1234)), np.vstack((allF,flist))
            
        numv = len(allV)-1
        mmm += num
        ck += 1
    sm = Mesh()
    sm.make_mesh(allV[1:],allF[1:])    
    return sm

#---------------------------------------------------------------------------
from scipy import sparse
from pypardiso import spsolve

def con_fair_midpoint0(c_v,c_vl,c_vr,N,return_s=False):
    "vl+vr-2v = 0"
    num = int(len(c_v)/3)     
    arr = np.arange(3*num)
    one = np.ones(3*num)
    row = np.tile(arr,3)
    col = np.r_[c_v,c_vl,c_vr]
    data = np.r_[2*one,-one,-one]
    K = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
    if return_s:
        return K, np.zeros(3*num)
    return K

def fair_vertices_or_vectors(Vertices,itera=10,efair=0.005,ee=0.001,is_fix=False):
    """
    given vertices(vectors)
    efair: fairness weight
    return same type, but after fairness-optimization
    """
    def matrix_fair(iva,ivb,ivc,num,var,efair):
        """midpoint: 2Q2 = Q1+Q3;
        """
        c_va = np.r_[iva, num+iva, 2*num+iva]
        c_vb = np.r_[ivb, num+ivb, 2*num+ivb]
        c_vc = np.r_[ivc, num+ivc, 2*num+ivc]
        K = con_fair_midpoint0(c_vb,c_va,c_vc,var)
        return efair * K    
    X = Vertices.flatten('F')
    num,var = len(Vertices),len(X)
    iva=np.arange(num-2)
    ivb,ivc = iva+1, iva+2
    K = matrix_fair(iva,ivb,ivc,num,var,efair)
    I = sparse.eye(var,format='coo')*ee**2
    r = np.zeros(K.shape[0])
    if is_fix:
        "fix two endpoint"
        vfix = np.r_[Vertices[0],Vertices[-1]] ##[x,y,z,x,y,z]
        col = np.array([0,num,2*num,num-1,2*num-1,3*num-1],dtype=int)
        data = np.ones(6)
        row = np.arange(6)
        F = sparse.coo_matrix((data,(row,col)), shape=(6, var)) 
        K = sparse.vstack((K,F*10))
        r = np.r_[r,vfix*10]
    n = 0
    opt_num = 100
    while n < itera and opt_num>1e-7 and opt_num<1e+6:
        X = spsolve(K.T*K+I, K.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
        n += 1
        opt_num = np.sum(np.square((K*X)))
    #print('fair-vectors:',n, '%.2g' %opt_num)
    return X.reshape(-1,3,order='F')