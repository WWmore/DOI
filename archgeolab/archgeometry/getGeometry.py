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