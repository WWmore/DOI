# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 22:16:21 2022

@author: WANGH0M
"""
__author__ = 'Hui'
#------------------------------------------------------------------------------
import numpy as np

from scipy import sparse
#------------------------------------------------------------------------------
from archgeolab.constraints.constraints_basic import column3D,con_edge,\
    con_unit,con_constl,con_equal_length,con_symmetry,\
    con_planarity,con_unit_normal,con_orient,con_diagonal,\
    con_equal_opposite_angle,con_dependent_vector
# -------------------------------------------------------------------------
"""
from archgeolab.constraints.constraints_net import 
    con_unit_edge,con_orient_rr_vn,con_orthogonal_midline,\
    con_anet,con_anet_diagnet,con_snet,con_snet_diagnet,\
    con_gnet,con_gnet_diagnet
    con_doi
        
"""

    #--------------------------------------------------------------------------
    #                Unit Edge Vectors, Unit Orient Normals:
    #-------------------------------------------------------------------------- 

def con_unit_edge(rregular=False,**kwargs): 
    """ unit_edge / unit_diag_edge_vec
    X += [l1,l2,l3,l4,ue1,ue2,ue3,ue4]; exists multiples between ue1,ue2,ue3,ue4
    (vi-v) = li*ui, ui**2=1, (i=1,2,3,4)
    """
    if kwargs.get('unit_diag_edge_vec'):
        w = kwargs.get('unit_diag_edge_vec')
        diag=True   
    elif kwargs.get('unit_edge_vec'):
        w = kwargs.get('unit_edge_vec')
        diag=False
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N5 = kwargs.get('N5')
    V = mesh.V
    if diag:
        v,v1,v2,v3,v4 = mesh.rr_star_corner 
    elif rregular:
        v,v1,v2,v3,v4 = mesh.rrv4f4
    else:
        #v,v1,v2,v3,v4 = mesh.ver_regular_star.T # default angle=90, non-orient
        v,v1,v2,v3,v4 = mesh.ver_star_matrix.T # oriented
    num = len(v)
    c_v = column3D(v,0,V)
    c_v1 = column3D(v1,0,V)
    c_v2 = column3D(v2,0,V)
    c_v3 = column3D(v3,0,V)
    c_v4 = column3D(v4,0,V)
    
    arr = np.arange(num)
    c_l1 = N5-16*num + arr
    c_l2 = c_l1 + num
    c_l3 = c_l2 + num
    c_l4 = c_l3 + num
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num)

    H1,r1 = con_edge(X,c_v1,c_v,c_l1,c_ue1)
    H2,r2 = con_edge(X,c_v2,c_v,c_l2,c_ue2)
    H3,r3 = con_edge(X,c_v3,c_v,c_l3,c_ue3)
    H4,r4 = con_edge(X,c_v4,c_v,c_l4,c_ue4)
    Hu1,ru1 = con_unit(X,c_ue1)
    Hu2,ru2 = con_unit(X,c_ue2)
    Hu3,ru3 = con_unit(X,c_ue3)
    Hu4,ru4 = con_unit(X,c_ue4)

    H = sparse.vstack((H1,H2,H3,H4,Hu1,Hu2,Hu3,Hu4))
    r = np.r_[r1,r2,r3,r4,ru1,ru2,ru3,ru4]
    
    #print('E1234:', np.sum(np.square((H*X)-r)))
    return H*w,r*w


def con_orient_rr_vn(**kwargs):
    """ X +=[vN, a], given computed Nv,which is defined at rr_vs
        vN *(e1-e3) = vN *(e2-e4) = 0; vN^2=1
        vN*Nv=a^2
    ==> vN is unit vertex-normal defined by t1xt2, **orient same with Nv**
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N5 = kwargs.get('N5')
    Norient = kwargs.get('Norient')
    v = mesh.ver_rrv4f4
    Nv = mesh.vertex_normals()[v]
    num = mesh.num_rrv4f4
    arr,arr3 = np.arange(num),np.arange(3*num)
    c_n = arr3 + Norient-4*num
    c_a = arr + Norient-num
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num)
    "vN should be oriented with oriented-vertex-normal"
    Hvn,rvn = con_unit_normal(X,c_ue1,c_ue2,c_ue3,c_ue4,c_n)
    
    "make sure the variable vN has same orientation with Nv:"
    Ho,ro = con_orient(X,Nv,c_n,c_a,neg=False)
    H = sparse.vstack((Hvn,Ho))
    r = np.r_[rvn,ro]
    return H,r


    #--------------------------------------------------------------------------
    #                       Orthogonal net:
    #-------------------------------------------------------------------------- 
def con_orthogonal_midline(is_rr=True,**kwargs): 
    """ 
    control quadfaces: two middle line are orthogonal to each other
    quadface: v1,v2,v3,v4 
    middle lins: e1 = (v1+v2)/2-(v3+v4)/2; e2 = (v2+v3)/2-(v4+v1)/2
    <===> e1 * e2 = 0 <==> (v1-v3)^2=(v2-v4)^2
    """
    w = kwargs.get('orthogonal')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
    if is_rr:
        ind = mesh.ind_rr_quadface_with_rrv
        v1,v2,v3,v4 = v1[ind],v2[ind],v3[ind],v4[ind]
    c_v1 = column3D(v1,0,mesh.V)
    c_v2 = column3D(v2,0,mesh.V)
    c_v3 = column3D(v3,0,mesh.V)
    c_v4 = column3D(v4,0,mesh.V)
    H,r = con_equal_length(X,c_v1,c_v2,c_v3,c_v4)
    return H*w,r*w 


    #--------------------------------------------------------------------------
    #           Geodesic parallel coordinates, SIR,  Kite:
    #--------------------------------------------------------------------------  
def con_doi(is_GO_or_OG=True,is_SIR=False,is_diagKite=False,**kwargs):
    """DOI-net: Discrete Orthogonal Isoceles Net 
                parametrizes Geodesic Paralel Coordinates
    
    parallel strip:            
                    v1 --- v4 --- v5 --- ...
                    |       |     |
                    v2 --- v3 --- v6 ---...
                
    orthogonal: based on each quad face, equal diagonal lengths
                (v1-v3)^2 = (v2-v4)^2
    parallel: equal geodesic-segment lengths along each parallel strip
                (v1-v2)^2=(v3-v4)^2=(v5-v6)^2=....
                
    SIR-net: Surface Isometric to surface of Revolution, based on DOI-net
    DOI + uniform edge lengths along each parallel       
                (v2-v3)^2=(v3-v6)^2=....
    
    Kite_net in diagonal: based on SIR-net, equal diagonals along parallel strip
    
    except all vertices vi, no extra variables
    """
    w = kwargs.get('DOI')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V

    def _orthogonal():
        v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
        if True:
            ind = mesh.ind_rr_quadface_with_rrv
            v1,v2,v3,v4 = v1[ind],v2[ind],v3[ind],v4[ind]
        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)
        H,r = con_equal_length(X,c_v1,c_v2,c_v3,c_v4)
        return H,r
    
    def _parallel():
        "regular patch / rotational patch"
        M = mesh.vMatrix if is_GO_or_OG else mesh.vMatrix.T
        
        "geodesic strips in vMatrix-vertical-direction, |v1-v2|=|v3-v4|"
        v1 = M[:-1,:-1].flatten('F') ## [upper row, left column]
        v2 = M[1:, :-1].flatten('F') ## [lower row, left column]
        v3 = M[1:, 1:].flatten('F')  ## [lower row, rigt column]
        v4 = M[:-1,1:].flatten('F')  ## [upper row, rigt column]

        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)

        H,r = con_equal_length(X,c_v1,c_v3,c_v2,c_v4)## (v1-v2)^2=(v3-v4)^2
        
        if is_SIR:
            "geodesic segment in vMatrix-horizontal-direction, |vl-vc|=|vc-vr|"
            vl = M[:, :-2].flatten('F') ## [all row, left column]
            vc = M[:,1:-1].flatten('F') ## [all row, cter column]
            vr = M[:,  2:].flatten('F')  ## [all row, rigt column]
            
            c_vl = column3D(vl,0,V)
            c_vc = column3D(vc,0,V)
            c_vr = column3D(vr,0,V)

            Hs,rs = con_symmetry(X,c_vl,c_vc,c_vr)  
            H = sparse.vstack((H, Hs))
            r = np.r_[r, rs]  

        if is_diagKite:
            """ based on _orthgonal(): equal diagonals in each quad
            all the diagonals along each parallel strip are equal
            if is_SIR=True: (default case)
                alignable Kite-net on SIR-srf
            else: 
                alignable Kite-net on GPC-net
            
            |v1-v3|=|v4-v6|=...
            """
            v1 = M[:-1, :-2].flatten('F')
            v3 = M[1:, 1:-1].flatten('F')
            v4 = M[:-1,1:-1].flatten('F')
            v6 = M[1:,   2:].flatten('F')

            c_v1 = column3D(v1,0,V)
            c_v3 = column3D(v3,0,V)
            c_v4 = column3D(v4,0,V)
            c_v6 = column3D(v6,0,V)

            Hk,rk = con_equal_length(X,c_v1,c_v4,c_v3,c_v6)## (v1-v3)^2=(v4-v6)^2
            H = sparse.vstack((H, Hk))
            r = np.r_[r, rk]   
            
        return H,r
    
    H1,r1 = _orthogonal()
    H2,r2 = _parallel()
            
    #print('ortho:', np.sum(np.square((H1*X)-r1)))
    #print('paral:', np.sum(np.square((H2*X)-r2)))
    
    H = sparse.vstack((H1, H2))
    r = np.r_[r1, r2]  
    return H*w,r*w

def con_doi__freeform(is_GO_or_OG=True,is_SIR=False,is_diagKite=False,**kwargs):
    """ above fucntion con_doi defined based on vMatrix from patch or rotational mesh
    can not handle the case with unregular boundaries
    this function should work on quad mesh with even singularieis (need check)
    Note: need check the orientation of equal edges
    """
    w = kwargs.get('DOI')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    
    def con_general_chebyshev(rhombus=False,half1=False,half2=False,is_rr=False):
        "each quadface, opposite edgelength equal"
        v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
        
        if is_rr:
            ind = mesh.ind_rr_quadface_with_rrv
            v1,v2,v3,v4 = v1[ind],v2[ind],v3[ind],v4[ind]
            
        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)
        if half1:
            H,r = con_equal_length(X,c_v1,c_v3,c_v2,c_v4)## (1-2)^2=(3-4)^2
        elif half2:
            H,r = con_equal_length(X,c_v1,c_v2,c_v4,c_v3)## (1-4)^2=(2-3)^2
        else:
            H1,r1 = con_equal_length(X,c_v1,c_v3,c_v2,c_v4)## (1-2)^2=(3-4)^2
            H2,r2 = con_equal_length(X,c_v1,c_v2,c_v4,c_v3)## (1-4)^2=(2-3)^2
            H = sparse.vstack((H1, H2))
            r = np.r_[r1,r2]
            if rhombus:
                "all edges are equal"
                H3,r3 = con_equal_length(X,c_v1,c_v2,c_v2,c_v3)## (1-2)^2=(2-3)^2
                H = sparse.vstack((H, H3))
                r = np.r_[r,r3]
        return H,r

    def con_equal_polysegment(is_poly1_or_2=True):
        "along one family of polylines, the segments are equal"
        v,v1,v2,v3,v4 = mesh.rrv4f4
        c_v = column3D(v,0,V)
        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)
        if is_poly1_or_2:
            "(v-v1)^2=(v-v3)^2 <==> v1^2 - v3^2 -2vv1 + 2vv3 = 0"
            c_vl, c_vr = c_v1, c_v3
        else:
            "(v-v2)^2=(v-v4)^2 <==> v2^2 - v4^2 -2vv2 + 2vv4 = 0"
            c_vl, c_vr = c_v2, c_v4
        H,r = con_symmetry(X,c_vl,c_v,c_vr)  
        return H,r

    H,r = con_general_chebyshev(half1=is_GO_or_OG,half2= not is_GO_or_OG)
    
    if is_SIR:
        Hs,rs = con_equal_polysegment(is_GO_or_OG)
        H = sparse.vstack((H, Hs))
        r = np.r_[r, rs]  
    
    if is_diagKite:
        """
        vertex star' 4faces' 4 corner vertex [a,b,c,d]
           a   1    d
           2   v    4
           b   3    c
        """
        v,va,vb,vc,vd = mesh.rr_star_corner
        c_v = column3D(v,0,V)
        c_va = column3D(va,0,V)
        c_vb = column3D(vb,0,V)
        c_vc = column3D(vc,0,V)
        c_vd = column3D(vd,0,V)
        H1,r1 = con_symmetry(X,c_va,c_v,c_vd)  
        H2,r2 = con_symmetry(X,c_vb,c_v,c_vc)  
        H = sparse.vstack((H, H1, H2))
        r = np.r_[r, r1, r2]   
    return H*w,r*w
    
def con_kite(is_transpose=False,is_diagGPC=False,is_diagSIR=False,is_rr=False,**kwargs):
    """Kite-net: two pairs of equal edge lengths within quad and on vertex-star

    is_diagGPC: 
    uniform symmetric-diagonal lengths along non-symmetric-diagonal polylines
    
    is_diagSIR (based on is_diagGPC):
    + uniform non-symmetric-diagonal lengths along non-symmetric-diagonal polylines    
    
    Kite-net quad face: (Hui: need check orientaion)
            v1      v7
        v2     v4        v6
        
            v3      v5

    Kite-net vertex star (Hui: choose this one):
           a   1    d
           2   v    4
           b   3    c  
    """
    w = kwargs.get('Kite')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    
    v,v1,v2,v3,v4 = mesh.rr_star.T
    if is_transpose:
        v1,v2,v3,v4 = v2,v3,v4,v1
        
    c_v = column3D(v,0,V)
    c_v1 = column3D(v1,0,V)
    c_v2 = column3D(v2,0,V)
    c_v3 = column3D(v3,0,V)
    c_v4 = column3D(v4,0,V)
       
    if False: ##need check the quad-vertex oriented samely as rr-vs
        "based on each quad face: |v1-v2|=|v1-v4|, |v3-v2|=|v3-v4|"
        v1,v2,v3,v4 = mesh.rr_quadface.T
        if is_transpose:
            v1,v2,v3,v4 = v2,v3,v4,v1
        
        if is_rr:
            ind = mesh.ind_rr_quadface_with_rrv
            v1,v2,v3,v4 = v1[ind],v2[ind],v3[ind],v4[ind]
            
        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)
        
        H1,r1 = con_symmetry(X,c_v2,c_v1,c_v4)  
        H2,r2 = con_symmetry(X,c_v2,c_v3,c_v4)  
        H = sparse.vstack((H1, H2))
        r = np.r_[r1, r2]  
    else:
        "based on vertex star: |v1-v|=|v2-v|, |v3-v|=|v4-v|"
        H1,r1 = con_symmetry(X,c_v1,c_v,c_v2)  
        H2,r2 = con_symmetry(X,c_v3,c_v,c_v4)  
        H = sparse.vstack((H1, H2))
        r = np.r_[r1, r2]  
    
        if is_diagGPC:
            "based on vertex star: |v2-v3|=|v1-v4|"
            Hg,rg = con_equal_length(X,c_v1,c_v2,c_v4,c_v3)
            H = sparse.vstack((H, Hg))
            r = np.r_[r, rg] 
            
            if is_diagSIR:
                "based on vertex star: |vb-v|=|vd-v|"
                v,va,vb,vc,vd = mesh.rr_star_corner
                if is_transpose:
                    va,vb,vc,vd = vb,vc,vd,va
                
                c_v = column3D(v,0,V)
                #c_va = column3D(va,0,V)
                c_vb = column3D(vb,0,V)
                #c_vc = column3D(vc,0,V)
                c_vd = column3D(vd,0,V)

                Hs,rs = con_symmetry(X,c_vb,c_v,c_vd)  
                H = sparse.vstack((H, Hs))
                r = np.r_[r, rs] 
        
    return H*w,r*w

def con_kite_diagnet(is_transpose=False,**kwargs):
    """
    Kite-diagnet: based on each vertex star, two pairs of equal diagonal lengths

    vertex star' 4faces' 4 corner vertex [a,b,c,d]:
           a   1    d
           2   v    4
           b   3    c  
           
    Kite_diagnet: |va-v|=|vd-v|, |vb-v|=|vc-v| & 
    (more stronger, but include boundary faces) |v1-v2|=|v1-v4|, |v2-v3|=|v4-v3|
    """
    w = kwargs.get('Kite_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V

    v,va,vb,vc,vd = mesh.rr_star_corner
    c_v = column3D(v,0,V)
    c_va = column3D(va,0,V)
    c_vb = column3D(vb,0,V)
    c_vc = column3D(vc,0,V)
    c_vd = column3D(vd,0,V)
    if is_transpose:
        H1,r1 = con_symmetry(X,c_va,c_v,c_vb)  
        H2,r2 = con_symmetry(X,c_vc,c_v,c_vd)  
    else:
        H1,r1 = con_symmetry(X,c_va,c_v,c_vd)  
        H2,r2 = con_symmetry(X,c_vb,c_v,c_vc)  
    H = sparse.vstack((H1, H2))
    r = np.r_[r1, r2]   
    if True:
        "more stronger: |v1-v2|=|v1-v4|, |v3-v2|=|v3-v4|"
        _,v1,v2,v3,v4 = mesh.rr_star.T
        c_v1 = column3D(v1,0,V)
        c_v2 = column3D(v2,0,V)
        c_v3 = column3D(v3,0,V)
        c_v4 = column3D(v4,0,V)
        if is_transpose:
            H1,r1 = con_symmetry(X,c_v1,c_v2,c_v3)  
            H2,r2 = con_symmetry(X,c_v1,c_v4,c_v3)  
        else:
            H1,r1 = con_symmetry(X,c_v2,c_v1,c_v4)  
            H2,r2 = con_symmetry(X,c_v2,c_v3,c_v4)  
        H = sparse.vstack((H, H1, H2))
        r = np.r_[r, r1, r2]   
    
    return H*w,r*w

    
def con_CGC(is_diag=False,is_rrvstar=False,**kwargs):
    """CGC_net: net curves of constant geodesic curvature, kg1=|kg2|=const.

    """
    w = kwargs.get('CGC')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    
    if is_rrvstar:
        if is_diag:
            w = kwargs.get('CGC_diagnet')
            v0,v1,v2,v3,v4 = mesh.rr_star_corner
        else:
            v0,v1,v2,v3,v4 = mesh.rrv4f4
    else:
        v0,v1,v2,v3,v4 = mesh.rr_star.T
    numv = len(v0) ##print(numv,mesh.num_rrv4f4)
    c_v0 = column3D(v0,0,V)
    c_v1 = column3D(v1,0,V)
    c_v2 = column3D(v2,0,V)
    c_v3 = column3D(v3,0,V)
    c_v4 = column3D(v4,0,V)
    arr1 = np.arange(numv)
    arr3 = np.arange(3*numv)   
    
    
def con_CNC(is_rr=False,**kwargs):
    """CNC_net: net curves of constant normal curvature, kn1=kn2=const.
        = S-net + constant radius 

    """
    
def con_Pnet(is_diag=False,is_rrvstar=False,**kwargs):
    """Pnet: pseudo-geodesic net: kg/kn=const.
    """

def con_gonet(rregular=False,is_direction24=False,**kwargs):
    """ paper: <Discrete GEODESIC PARALLEL COORDINATES>-SIGGRAPH ASIA 2019
    based on con_unit_edge() & con_1geodesic
    orthogonal: (e1-e3)*(e2-e4) = 0
    tangents := e1-e3, e2-e4
    normal := t1 x t2
    if direction: 
        geodesic: e1*e2-e1*e4=0;  e2*e3-e3*e4=0; 
    else:
        geodesic: e1*e2-e2*e3=0;  e3*e4-e4*e1=0;
    """
    w = kwargs.get('GOnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    if rregular:
        num=mesh.num_rrv4f4
    else:
        num = mesh.num_regular
        
    arr = np.arange(num)
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num) 
  
    if is_direction24:
        H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue2,c_ue3)
        H2,r2 = con_equal_opposite_angle(X,c_ue3,c_ue4,c_ue4,c_ue1)
    else:
        H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue1,c_ue4)
        H2,r2 = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue3,c_ue4)
    
    H = sparse.vstack((H1, H2))
    r = np.r_[r1, r2]
    
    if 0:
        "additional orthogonal: (e1-e3)*(e2-e4) = 0"          
        row = np.tile(arr,12)
        col = np.r_[c_ue1,c_ue2,c_ue3,c_ue4]
        data = np.r_[X[c_ue2]-X[c_ue4],X[c_ue1]-X[c_ue3],X[c_ue4]-X[c_ue2],X[c_ue3]-X[c_ue1]]
        H3 = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        r3 = np.einsum('ij,ij->i',(X[c_ue1]-X[c_ue3]).reshape(-1,3, order='F'),(X[c_ue2]-X[c_ue4]).reshape(-1,3, order='F'))
        #print(H1.shape,H2.shape,H3.shape,r1.shape,r2.shape,r3.shape)
        H = sparse.vstack((H1, H2, H3))
        r = np.r_[r1, r2, r3]  
        
    #print('gonet:',np.sum(np.square(H*X-r)))
    return H*w,r*w

def con_dgpc(rregular=False,polyline_direction=False,**kwargs):
    """main difference here is using patch_matrix to represent all vertices
    based on con_unit_edge() & con_gonet
    equal-geodesic-segment lengths along parallel-direction
    each row: (vi-vj)^2 - lij^2 = 0
    """    
    w = kwargs.get('DGPC')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    
    Ndgpc = kwargs.get('Ndgpc')
    
    rm = mesh.patch_matrix
    if polyline_direction:
        rm = rm.T
    nrow,ncol = rm.shape    
    vi,vj = rm[:,:-1].flatten(), rm[:,1:].flatten()
    c_vi = column3D(vi ,0,mesh.V)
    c_vj = column3D(vj ,0,mesh.V)
    c_l = (Ndgpc-nrow+np.arange(nrow)).repeat(ncol-1)
    H,r = con_diagonal(X,c_vi,c_vj,c_l,nrow*(ncol-1))
    return H*w,r*w

    #--------------------------------------------------------------------------
    #                       A-net:
    #-------------------------------------------------------------------------- 
def _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4):
    "vn*(vi-v)=0; vn**2=1"
    H1,r1 = con_planarity(X,c_v,c_v1,c_n)
    H2,r2 = con_planarity(X,c_v,c_v2,c_n)
    H3,r3 = con_planarity(X,c_v,c_v3,c_n)
    H4,r4 = con_planarity(X,c_v,c_v4,c_n)
    Hn,rn = con_unit(X,c_n)
    H = sparse.vstack((H1,H2,H3,H4,Hn))
    r = np.r_[r1,r2,r3,r4,rn]
    return H*w, r*w
    
def con_anet(rregular=False,**kwargs):
    """ based on con_unit_edge()
    X += [ni]
    ni * (vij - vi) = 0
    """
    w = kwargs.get('Anet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    
    Nanet = kwargs.get('Nanet')
    
    if rregular:
        v,v1,v2,v3,v4 = mesh.rrv4f4
        num=mesh.num_rrv4f4
    else:
        num = mesh.num_regular
        v,v1,v2,v3,v4 = mesh.ver_regular_star.T
        
    c_n = Nanet-3*num+np.arange(3*num)
    c_v  = column3D(v ,0,mesh.V)
    c_v1 = column3D(v1,0,mesh.V)
    c_v2 = column3D(v2,0,mesh.V)
    c_v3 = column3D(v3,0,mesh.V)
    c_v4 = column3D(v4,0,mesh.V)
    
    H,r = _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4)
    return H,r
     
def con_anet_diagnet(**kwargs):
    "based on con_unit_edge(diag=True); X += [ni]; ni * (vij - vi) = 0"
    w = kwargs.get('Anet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    Nanet = kwargs.get('Nanet')
    
    #c_v,c_v1,c_v2,c_v3,c_v4 = mesh.get_vs_diagonal_v(index=False)
    v,v1,v2,v3,v4 = mesh.rr_star_corner
    c_v  = column3D(v ,0,mesh.V)
    c_v1 = column3D(v1,0,mesh.V)
    c_v2 = column3D(v2,0,mesh.V)
    c_v3 = column3D(v3,0,mesh.V)
    c_v4 = column3D(v4,0,mesh.V)
    
    num = int(len(c_v)/3)
    c_n = Nanet-3*num+np.arange(3*num)

    H,r = _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4)
    return H,r


    #--------------------------------------------------------------------------
    #                       S-net:
    #--------------------------------------------------------------------------  
def con_snet(orientrn,is_rrvstar=True,is_diagmesh=False,
             is_uniqR=False,assigned_r=None,**kwargs):
    """a(x^2+y^2+z^2)+(bx+cy+dz)+e=0 ; normalize: F^2 = b^2+c^2+d^2-4ae=1
    sphere center C:= (m1,m2,m3) = -(b, c, d) /a/2
    sphere radius:= F /a/2
    unit_sphere_normal N==-(2*A*Vx+B, 2*A*Vy+C, 2*A*Vz+D), (pointing from v to center)
    since P(Vx,Vy,Vz) satisfy the sphere eq. and the normalizated eq., so that
        N ^2=1
    """
    w = kwargs.get('Snet')
    Nsnet = kwargs.get('Nsnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V

    if is_rrvstar:
        "new for SSGweb-project"
        if is_diagmesh:
            w = kwargs.get('Snet_diagnet')
            v0,v1,v2,v3,v4 = mesh.rr_star_corner
        else:
            v0,v1,v2,v3,v4 = mesh.rrv4f4
        #orientrn = orientrn[mesh.ind_rr_star_v4f4] ##below should be the same
        ##print(len(mesh.ind_rr_star_v4f4),len(v0))
    else:
        "used in CRPC"
        v0,v1,v2,v3,v4 = mesh.rr_star.T
    numv = len(v0) ##print(numv,mesh.num_rrv4f4)
    c_v0 = column3D(v0,0,V)
    c_v1 = column3D(v1,0,V)
    c_v2 = column3D(v2,0,V)
    c_v3 = column3D(v3,0,V)
    c_v4 = column3D(v4,0,V)
    arr1 = np.arange(numv)
    arr3 = np.arange(3*numv)
    _n1 = Nsnet-11*numv
    c_squ, c_a = _n1+np.arange(5*numv),_n1+5*numv+arr1
    c_b,c_c,c_d,c_e = c_a+numv,c_a+2*numv,c_a+3*numv,c_a+4*numv
    c_a_sqr = c_a+5*numv

    def _con_v_square(c_squ):
        "[v;v1,v2,v3,v4]=[x,y,z], X[c_squ]=x^2+y^2+z^2"
        row_v = np.tile(arr1,3)
        row_1 = row_v+numv
        row_2 = row_v+2*numv
        row_3 = row_v+3*numv
        row_4 = row_v+4*numv
        row = np.r_[row_v,row_1,row_2,row_3,row_4,np.arange(5*numv)]
        col = np.r_[c_v0,c_v1,c_v2,c_v3,c_v4,c_squ]
        dv = 2*np.r_[X[c_v0]]
        d1 = 2*np.r_[X[c_v1]]
        d2 = 2*np.r_[X[c_v2]]
        d3 = 2*np.r_[X[c_v3]]
        d4 = 2*np.r_[X[c_v4]]
        data = np.r_[dv,d1,d2,d3,d4,-np.ones(5*numv)]
        H = sparse.coo_matrix((data,(row,col)), shape=(5*numv, N))
        def xyz(c_i):
            c_x = c_i[:numv]
            c_y = c_i[numv:2*numv]
            c_z = c_i[2*numv:]
            return np.r_[X[c_x]**2+X[c_y]**2+X[c_z]**2]
        r = np.r_[xyz(c_v0),xyz(c_v1),xyz(c_v2),xyz(c_v3),xyz(c_v4)]
        return H,r
    def _con_pos_a(c_a,c_a_sqr):
        "a>=0 <---> a_sqr^2 - a = 0"
        row = np.tile(arr1,2)
        col = np.r_[c_a_sqr, c_a]
        data = np.r_[2*X[c_a_sqr], -np.ones(numv)]
        r = X[c_a_sqr]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e):
        """normalize the sphere equation,
        convinent for computing/represent distance\normals
        ||df|| = b^2+c^2+d^2-4ae=1
        """
        row = np.tile(arr1,5)
        col = np.r_[c_a,c_b,c_c,c_d,c_e]
        data = 2*np.r_[-2*X[c_e],X[c_b],X[c_c],X[c_d],-2*X[c_a]]
        r = X[c_b]**2+X[c_c]**2+X[c_d]**2-4*X[c_a]*X[c_e]+np.ones(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere(c_squ,c_a,c_b,c_c,c_d,c_e):
        "a(x^2+y^2+z^2)+(bx+cy+dz)+e=0"
        row = np.tile(arr1,9)
        def __sphere(c_vi,c_sq):
            c_x = c_vi[:numv]
            c_y = c_vi[numv:2*numv]
            c_z = c_vi[2*numv:]
            col = np.r_[c_x,c_y,c_z,c_sq,c_a,c_b,c_c,c_d,c_e]
            data = np.r_[X[c_b],X[c_c],X[c_d],X[c_a],X[c_sq],X[c_x],X[c_y],X[c_z],np.ones(numv)]
            r = X[c_b]*X[c_x]+X[c_c]*X[c_y]+X[c_d]*X[c_z]+X[c_a]*X[c_sq]
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        H0,r0 = __sphere(c_v0,c_squ[:numv])
        H1,r1 = __sphere(c_v1,c_squ[numv:2*numv])
        H2,r2 = __sphere(c_v2,c_squ[2*numv:3*numv])
        H3,r3 = __sphere(c_v3,c_squ[3*numv:4*numv])
        H4,r4 = __sphere(c_v4,c_squ[4*numv:])
        H = sparse.vstack((H0,H1,H2,H3,H4))
        r = np.r_[r0,r1,r2,r3,r4]
        return H,r
    def _con_const_radius(c_a,c_r):
        "2*ai * r = 1 == df"
        c_rr = np.tile(c_r, numv)
        row = np.tile(arr1,2)
        col = np.r_[c_a, c_rr]
        data = np.r_[X[c_rr], X[c_a]]
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        r = X[c_rr] * X[c_a] + 0.5*np.ones(numv)
        return H,r
    def _con_anet(c_a):
        row = arr1
        col = c_a
        data = np.ones(numv)
        r = np.zeros(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    
    "this fun. is new added."
    def _con_unit_sphere_normal(c_v0,c_a,c_b,c_c,c_d,c_n): ##need definition of unitnormal
        "N==-(2*A*Vx+B, 2*A*Vy+C, 2*A*Vz+D)"
        c_nx,c_ny,c_nz = c_n[arr1], c_n[arr1+numv], c_n[arr1+2*numv]
        c_vx,c_vy,c_vz = c_v0[arr1], c_v0[arr1+numv], c_v0[arr1+2*numv]
        def _con_coordinate(c_nx,c_vx,c_a,c_b):
            "2*a*vx+b+nx = 0"
            col = np.r_[c_nx,c_vx,c_a,c_b]
            row = np.tile(arr1, 4)
            one = np.ones(numv)
            data = np.r_[one,2*X[c_a],2*X[c_vx],one]
            r = 2*X[c_a]*X[c_vx]
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        Hx,rx = _con_coordinate(c_nx,c_vx,c_a,c_b)
        Hy,ry = _con_coordinate(c_ny,c_vy,c_a,c_c)
        Hz,rz = _con_coordinate(c_nz,c_vz,c_a,c_d)
        H = sparse.vstack((Hx,Hy,Hz))
        r = np.r_[rx,ry,rz]
        return H,r
        
    def _con_orient(c_n,c_o):
        "N*orientn = const^2 <==> n0x*nx+n0y*ny+n0z*nz-x_orient^2 = 0"
        row = np.tile(arr1,4)
        col = np.r_[c_n, c_o]
        data = np.r_[orientrn.flatten('F'), -2*X[c_o]]
        r = -X[c_o]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        "add limitation for spherical normal"
        return H,r

    H0,r0 = _con_v_square(c_squ)
    H1,r1 = _con_pos_a(c_a,c_a_sqr)
    Hn,rn = _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e)
    Hs,rs = _con_sphere(c_squ,c_a,c_b,c_c,c_d,c_e)
    H = sparse.vstack((H0,H1,Hn,Hs))
    r = np.r_[r0,r1,rn,rs]
    # print('s0:', np.sum(np.square((H0*X)-r0)))
    # print('s1:', np.sum(np.square((H1*X)-r1)))
    # print('s2:', np.sum(np.square((Hn*X)-rn)))
    # print('s3:', np.sum(np.square((Hs*X)-rs)))
    # print('snet:', np.sum(np.square((H*X)-r)))
    if kwargs.get('Snet_orient'):
        w1 = kwargs.get('Snet_orient')
        Ns_n = kwargs.get('Ns_n')
        c_n = Ns_n-4*numv+arr3
        c_n_sqr = Ns_n-numv+arr1
        Ho,ro = _con_orient(c_n,c_n_sqr)
        Hn,rn = _con_unit_sphere_normal(c_v0,c_a,c_b,c_c,c_d,c_n)
        H = sparse.vstack((H, Ho * w1, Hn*w1*10)) #need check the weight
        r = np.r_[r, ro * w1, rn*w1*10]
        # print('o:', np.sum(np.square((Ho*X)-ro)))
        # print('n:', np.sum(np.square((Hn*X)-rn)))
        
    if kwargs.get('Snet_constR'):
        w2 = kwargs.get('Snet_constR')
        Ns_r = kwargs.get('Ns_r')
        c_r = np.array([Ns_r-1],dtype=int)
        Hr,rr = _con_const_radius(c_a,c_r)
        H = sparse.vstack((H, Hr * w2))
        r = np.r_[r, rr * w2]
        if is_uniqR:
            H0,r0 = con_constl(c_r,assigned_r,N)
            H = sparse.vstack((H, H0))
            r = np.r_[r,r0]
        #print('r:', np.sum(np.square((Hr*X)-rr)))
    if kwargs.get('Snet_anet'):
        w3 = kwargs.get('Snet_anet')
        Ha,ra = _con_anet(c_a)
        H = sparse.vstack((H, Ha * w3))
        r = np.r_[r, ra * w3]

    return H*w,r*w

def con_snet_diagnet(orientrn,**kwargs):
    w = kwargs.get('Snet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nsnet = kwargs.get('Nsnet')
    
    V = mesh.V
    numv = mesh.num_rrv4f4
    arr1 = np.arange(numv)
    arr3 = np.arange(3*numv)
    
    #c_v,c_cen1,c_cen2,c_cen3,c_cen4 = mesh.get_vs_diagonal_v(ck1=ck1,ck2=ck2,index=False)
    v,v1,v2,v3,v4 = mesh.rr_star_corner
    c_v = column3D(v,0,V)
    c_cen1 = column3D(v1,0,V)
    c_cen2 = column3D(v2,0,V)
    c_cen3 = column3D(v3,0,V)
    c_cen4 = column3D(v4,0,V)
    c_cen = [c_cen1,c_cen2,c_cen3,c_cen4]
    _n1 = Nsnet-11*numv
    c_squ, c_a = _n1+np.arange(5*numv),_n1+5*numv+arr1
    c_b,c_c,c_d,c_e = c_a+numv,c_a+2*numv,c_a+3*numv,c_a+4*numv
    c_a_sqr = c_a+5*numv

    def _con_v_square(c_v,c_cen,c_squ):
        "[v;c1,c2,c3,c4]=[x,y,z], X[c_squ]=x^2+y^2+z^2"
        c_cen1,c_cen2,c_cen3,c_cen4 = c_cen
        row_v = np.tile(arr1,3)
        row_1 = row_v+numv
        row_2 = row_v+2*numv
        row_3 = row_v+3*numv
        row_4 = row_v+4*numv
        row = np.r_[row_v,row_1,row_2,row_3,row_4,np.arange(5*numv)]
        col = np.r_[c_v,c_cen1,c_cen2,c_cen3,c_cen4,c_squ]
        dv = 2*np.r_[X[c_v]]
        d1 = 2*np.r_[X[c_cen1]]
        d2 = 2*np.r_[X[c_cen2]]
        d3 = 2*np.r_[X[c_cen3]]
        d4 = 2*np.r_[X[c_cen4]]
        data = np.r_[dv,d1,d2,d3,d4,-np.ones(5*numv)]
        H = sparse.coo_matrix((data,(row,col)), shape=(5*numv, N))
        def xyz(c_i):
            c_x = c_i[:numv]
            c_y = c_i[numv:2*numv]
            c_z = c_i[2*numv:]
            return np.r_[X[c_x]**2+X[c_y]**2+X[c_z]**2]
        r = np.r_[xyz(c_v),xyz(c_cen1),xyz(c_cen2),xyz(c_cen3),xyz(c_cen4)]
        return H,r
    
    def _con_pos_a(c_a,c_a_sqr):
        "a>=0 <---> a_sqr^2 - a = 0"
        row = np.tile(arr1,2)
        col = np.r_[c_a_sqr, c_a]
        data = np.r_[2*X[c_a_sqr], -np.ones(numv)]
        r = X[c_a_sqr]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    
    def _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e):
        """normalize the sphere equation,
        convinent for computing/represent distance\normals
        ||df|| = b^2+c^2+d^2-4ae=1
        """
        row = np.tile(arr1,5)
        col = np.r_[c_a,c_b,c_c,c_d,c_e]
        data = 2*np.r_[-2*X[c_e],X[c_b],X[c_c],X[c_d],-2*X[c_a]]
        r = X[c_b]**2+X[c_c]**2+X[c_d]**2-4*X[c_a]*X[c_e]+np.ones(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    
    def _con_sphere(c_v,c_cen,c_squ,c_a,c_b,c_c,c_d,c_e):
        "a(x^2+y^2+z^2)+(bx+cy+dz)+e=0"
        c_cen1,c_cen2,c_cen3,c_cen4 = c_cen
        row = np.tile(arr1,9)
        def __sphere(c_vi,c_sq):
            c_x = c_vi[:numv]
            c_y = c_vi[numv:2*numv]
            c_z = c_vi[2*numv:]
            col = np.r_[c_x,c_y,c_z,c_sq,c_a,c_b,c_c,c_d,c_e]
            data = np.r_[X[c_b],X[c_c],X[c_d],X[c_a],X[c_sq],X[c_x],X[c_y],X[c_z],np.ones(numv)]
            r = X[c_b]*X[c_x]+X[c_c]*X[c_y]+X[c_d]*X[c_z]+X[c_a]*X[c_sq]
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        H0,r0 = __sphere(c_v,c_squ[:numv])
        H1,r1 = __sphere(c_cen1,c_squ[numv:2*numv])
        H2,r2 = __sphere(c_cen2,c_squ[2*numv:3*numv])
        H3,r3 = __sphere(c_cen3,c_squ[3*numv:4*numv])
        H4,r4 = __sphere(c_cen4,c_squ[4*numv:])
        H = sparse.vstack((H0,H1,H2,H3,H4))
        r = np.r_[r0,r1,r2,r3,r4]
        return H,r   
    
    def _con_const_radius(c_a,c_r):
        "2*ai * r = 1 == df"
        c_rr = np.tile(c_r, numv)
        row = np.tile(arr1,2)
        col = np.r_[c_a, c_rr]
        data = np.r_[X[c_rr], X[c_a]]
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        r = X[c_rr] * X[c_a] + 0.5*np.ones(numv)
        return H,r
    
    def _con_orient(c_n,c_o):
        "n0x*nx+n0y*ny+n0z*nz-x_orient^2 = 0"
        row = np.tile(arr1,4)
        col = np.r_[c_n, c_o]
        data = np.r_[orientrn.flatten('F'), -2*X[c_o]]
        r = -X[c_o]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    
    def _con_unit_normal(c_n):
        cen = -np.c_[X[c_b]/X[c_a],X[c_c]/X[c_a],X[c_d]/X[c_a]]/2
        rad1 = np.linalg.norm(cen-X[c_v].reshape(-1,3,order='F'),axis=1)
        rad2 = np.linalg.norm(cen-X[c_cen1].reshape(-1,3,order='F'),axis=1)
        rad3 = np.linalg.norm(cen-X[c_cen2].reshape(-1,3,order='F'),axis=1)
        rad4 = np.linalg.norm(cen-X[c_cen3].reshape(-1,3,order='F'),axis=1)
        rad5 = np.linalg.norm(cen-X[c_cen4].reshape(-1,3,order='F'),axis=1)
        radii = (rad1+rad2+rad3+rad4+rad5)/5
        
        def _normal(c_a,c_b,c_anx,c_nx):
            row = np.tile(np.arange(numv),4)
            col = np.r_[c_a,c_b,c_anx,c_nx]
            one = np.ones(numv)
            data = np.r_[2*(radii*X[c_nx]+X[c_anx]),one,2*X[c_a],2*radii*X[c_a]]
            r = 2*X[c_a]*(radii*X[c_nx]+X[c_anx])
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        Hb,rb = _normal(c_a,c_b,c_v[:numv],c_n[:numv])
        Hc,rc = _normal(c_a,c_c,c_v[numv:2*numv],c_n[numv:2*numv])
        Hd,rd = _normal(c_a,c_d,c_v[2*numv:],c_n[2*numv:])
        Hn,rn = con_unit(X,c_n)
        H = sparse.vstack((Hb, Hc, Hd, Hn))
        r = np.r_[rb, rc, rd, rn]  
        return H,r    
    
    H0,r0 = _con_v_square(c_v,c_cen,c_squ)
    H1,r1 = _con_pos_a(c_a,c_a_sqr)
    Hn,rn = _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e)
    Hs,rs = _con_sphere(c_v,c_cen,c_squ,c_a,c_b,c_c,c_d,c_e)
    H = sparse.vstack((H0,H1,Hn,Hs))
    r = np.r_[r0,r1,rn,rs]
 
    if kwargs.get('Snet_orient'):
        w1 = kwargs.get('Snet_orient')
        Ns_n = kwargs.get('Ns_n')
        c_n = Ns_n-4*numv+arr3
        c_n_sqr = Ns_n-numv+arr1
        Ho,ro = _con_orient(c_n,c_n_sqr)
        H = sparse.vstack((H, Ho * w1))
        r = np.r_[r, ro * w1]
        ##c
    if kwargs.get('Snet_constR'):
        w2 = kwargs.get('Snet_constR')
        Ns_r = kwargs.get('Ns_r')
        c_r = np.array([Ns_r-1],dtype=int)
        Hr,rr = _con_const_radius(c_a,c_r)
        H = sparse.vstack((H, Hr * w2))
        r = np.r_[r, rr * w2]
 
    return H*w,r*w


    #--------------------------------------------------------------------------
    #                       G-net:
    #--------------------------------------------------------------------------  
def _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4):
    H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue3,c_ue4)
    H2,r2 = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue4,c_ue1)
    H, r = sparse.vstack((H1, H2)), np.r_[r1,r2]
    return H*w, r*w

def con_gnet(rregular=True,checker_weight=1,id_checker=None,**kwargs):
    """
    based on con_unit_edge(diag=False)
    e1*e2-e3*e4=0; e2*e3-e1*e4=0
    """
    w = kwargs.get('Gnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N5 = kwargs.get('N5')
    
    if rregular:
        "function same as below:con_gnet_diagnet"
        num=mesh.num_rrv4f4
    else:
        num = mesh.num_regular
    arr = np.arange(num)
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num)     
    
    if rregular and checker_weight<1:   ##no use
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = column3D(iblue,0,mesh.num_rrv4f4)
        ir = column3D(ired,0,mesh.num_rrv4f4)  
        Hb,rb = _con_gnet(X,w,c_ue1[ib],c_ue2[ib],c_ue3[ib],c_ue4[ib])
        Hr,rr = _con_gnet(X,wr,c_ue1[ir],c_ue2[ir],c_ue3[ir],c_ue4[ir])
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4)
    
    return H,r

def con_gnet_diagnet(checker_weight=1,id_checker=None,**kwargs):
    """
    based on con_unit_edge(diag=True)
    e1*e2-e3*e4=0; e2*e3-e1*e4=0
    """
    w = kwargs.get('Gnet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N5 = kwargs.get('N5')
    
    num = mesh.num_rrv4f4
    arr = np.arange(num)
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num)        
    
    if checker_weight<1:  ##no use
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = column3D(iblue,0,mesh.num_rrv4f4)
        ir = column3D(ired,0,mesh.num_rrv4f4)  
        Hb,rb = _con_gnet(X,w,c_ue1[ib],c_ue2[ib],c_ue3[ib],c_ue4[ib])
        Hr,rr = _con_gnet(X,wr,c_ue1[ir],c_ue2[ir],c_ue3[ir],c_ue4[ir])
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4)
    
    return H,r

    #--------------------------------------------------------------------------
    #                      rulings:
    #-------------------------------------------------------------------------- 

def con_polyline_ruling(switch_diagmeth=False,**kwargs):
    """ X +=[ni]
    along each i-th polyline: ti x (vij-vik) = 0; k=j+1,j=0,...
    refer: self._con_agnet_planar_geodesic(),self.get_poly_strip_ruling_tangent()
    """
    w = kwargs.get('ruling')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    Nruling = kwargs.get('Nruling')
    
    iall = mesh.get_both_isopolyline(diagpoly=switch_diagmeth) # interval is random
    num = len(iall)
    arr = Nruling-3*num+np.arange(3*num)
    c_tx,c_ty,c_tz = arr[:num],arr[num:2*num],arr[2*num:3*num]

    alla=allb = np.array([],dtype=int)
    alltx=allty=alltz = np.array([],dtype=int)
    i = 0
    for iv in iall:
        "t x (a-b) = 0"
        va,vb = iv[:-1],iv[1:]
        alla = np.r_[alla,va]
        allb = np.r_[allb,vb]
        m = len(va)
        alltx = np.r_[alltx,np.tile(c_tx[i],m)]
        allty = np.r_[allty,np.tile(c_ty[i],m)]
        alltz = np.r_[alltz,np.tile(c_tz[i],m)]
        i += 1
    c_a = column3D(alla,0,mesh.V)
    c_b = column3D(allb,0,mesh.V)
    c_ti = np.r_[alltx,allty,alltz]
    H,r = con_dependent_vector(X,c_a,c_b,c_ti)
    H1,r1 = con_unit(X,arr)
    H = sparse.vstack((H,H1))
    r = np.r_[r,r1]
    #self.add_iterative_constraint(H * w, r * w, 'ruling')    
    return H*w, r*w
 