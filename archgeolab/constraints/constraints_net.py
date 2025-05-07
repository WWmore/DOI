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
    con_planarity,con_unit_normal,con_diagonal,con_osculating_tangent,\
    con_equal_opposite_angle,con_dependent_vector,con_cross,\
    con_constangle2,con_constangle3,con_constangle4,con_positive,con_negative,\
    con_orient,con_orient1,con_orient2,con_ortho,con_orthogonal_2vectors,\
    con_diagonal2,con_circle
# -------------------------------------------------------------------------
"""
from archgeolab.constraints.constraints_net import 
    con_unit_edge,con_orient_rr_vn,con_orthogonal_midline,\
    con_anet,con_anet_diagnet,con_snet,con_gnet,
    con_doi
        
"""

    #--------------------------------------------------------------------------
    #                Unit Edge Vectors, Unit Orient Normals:
    #-------------------------------------------------------------------------- 

def con_unit_edge(is_diagnet=False,rregular=True,**kwargs): 
    """ unit_edge / unit_diag_edge_vec
    X += [l1,l2,l3,l4,ue1,ue2,ue3,ue4]; exists multiples between ue1,ue2,ue3,ue4
    (vi-v) = li*ui, ui**2=1, (i=1,2,3,4)
    """
    diag = True if is_diagnet else False
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
    return H, r

def con_osculating_tangents(is_diagnet=False,**kwargs):
    """X +=[ll1,ll2,ll3,ll4,lt1,lt2,t1,t2]  (defined at rrv4f4)
    t1,t2 are built from con_osculating_tangent
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    Noscut = kwargs.get('Noscut')
    
    if is_diagnet:
        v,va,vb,vc,vd = mesh.rr_star_corner# in diagonal direction
        c_v = column3D(v,0,mesh.V)
        c_1 = column3D(va,0,mesh.V)
        c_2 = column3D(vb,0,mesh.V)
        c_3 = column3D(vc,0,mesh.V)
        c_4 = column3D(vd,0,mesh.V)
    else:
        v,v1,v2,v3,v4 = mesh.rrv4f4
        c_v = column3D(v,0,mesh.V)
        c_1 = column3D(v1,0,mesh.V)
        c_2 = column3D(v2,0,mesh.V)
        c_3 = column3D(v3,0,mesh.V)
        c_4 = column3D(v4,0,mesh.V)         
    num = len(v)
    arr,arr3 = np.arange(num),np.arange(3*num)
    s = Noscut - 12*num
    c_ll1 = s+arr
    c_ll2,c_ll3,c_ll4 = c_ll1+num, c_ll1+2*num, c_ll1+3*num
    c_lt1,c_lt2 = c_ll1+4*num, c_ll1+5*num
    c_t1,c_t2 = s+6*num+arr3, s+9*num+arr3
    H1,r1 = con_osculating_tangent(X,c_v,c_1,c_3,c_ll1,c_ll3,c_lt1,c_t1,num)
    H2,r2 = con_osculating_tangent(X,c_v,c_2,c_4,c_ll2,c_ll4,c_lt2,c_t2,num)
    H = sparse.vstack((H1,H2))
    r = np.r_[r1,r2]
    return H,r

def con_orient_rr_vn(is_osculating_tangent=False,**kwargs):
    """ X +=[vN, a], given computed Nv,which is defined at rr_vs
    Default: 
        vN *(e1-e3) = vN *(e2-e4) = 0; vN^2=1
        vN*Nv=a^2
    elif is_osculating_tangent:
        X += [vN,a]
        vN * t1 = vN * t2 = 0; vN^2=1  <==> vN = t1 x t2
        vN*Nv=a^2
    ==> vN is unit vertex-normal defined by t1xt2, **orient same with Nv**
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N5 = kwargs.get('N5')
    Norient = kwargs.get('Norient')
    v = mesh.ver_rrv4f4
    Nv = mesh.vertex_normals()[v]
    num = len(v) ##= mesh.num_rrv4f4
    arr,arr3 = np.arange(num),np.arange(3*num)
    c_n = arr3 + Norient-4*num
    c_a = arr + Norient-num
    
    if is_osculating_tangent:
        s = kwargs.get('Noscut')- 12*num
        c_t1,c_t2 = s+6*num+arr3, s+9*num+arr3
        Hvn,rvn = con_cross(X,c_t1,c_t2,c_n)
    else:
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
    
    is_transpose: |va-v|=|vb-v|, |vc-v|=|vd-v| & |v2-v1|=|v2-v3|, |v4-v1|=|v4-v3|
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
    #                      CGC / G-net:
    #--------------------------------------------------------------------------  
def con_CGC(is_diagnet=False,is_rrvstar=False,**kwargs):
    """CGC_net: net curves of constant geodesic curvature, kg1=|kg2|=const.
    
    """
    w = kwargs.get('CGC')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    
    if is_rrvstar:
        if is_diagnet:
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
    
    return H,r


def con_gnet(rregular=True,**kwargs):
    """
    Gnet: based on con_unit_edge(diag=False)
    Gnet_diagnet: based on con_unit_edge(diag=True)
    e1*e2-e3*e4=0; e2*e3-e1*e4=0
    """
    def _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4):
        H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue3,c_ue4)
        H2,r2 = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue4,c_ue1)
        H, r = sparse.vstack((H1, H2)), np.r_[r1,r2]
        return H*w, r*w
    
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
    H,r = _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4)
    return H,r


    #--------------------------------------------------------------------------
    #                       CNC / S-net / Anet :
    #--------------------------------------------------------------------------  
    
def con_CNC(is_rr=False,**kwargs):
    """CNC_net: net curves of constant normal curvature, kn1=kn2=const.
        = S-net + constant radius 
    refer to : con_snet(is_uniqR=True)
    """
def con_snet(orientrn,is_rrvstar=True,is_diagnet=False,
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

    if is_rrvstar:  ##default=True
        "new for SSGweb-project"
        if is_diagnet:
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


def con_anet(rregular=False,is_diagnet=False,**kwargs):
    """ based on con_unit_edge()
    X += [ni]
    ni * (vij - vi) = 0
    """
    w = kwargs.get('Anet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    
    Nanet = kwargs.get('Nanet')
    
    if is_diagnet:
        "based on con_unit_edge(diag=True); X += [ni]; ni * (vij - vi) = 0"
        v,v1,v2,v3,v4 = mesh.rr_star_corner
    else:
        if rregular:
            v,v1,v2,v3,v4 = mesh.rrv4f4
            #num=mesh.num_rrv4f4
        else:
            #num = mesh.num_regular
            v,v1,v2,v3,v4 = mesh.ver_regular_star.T
    
    num = len(v)
    c_n = Nanet-3*num+np.arange(3*num)
    c_v  = column3D(v ,0,mesh.V)
    c_v1 = column3D(v1,0,mesh.V)
    c_v2 = column3D(v2,0,mesh.V)
    c_v3 = column3D(v3,0,mesh.V)
    c_v4 = column3D(v4,0,mesh.V)
    
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
    
    H,r = _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4)
    return H,r

    #--------------------------------------------------------------------------
    #                       P-net:
    #--------------------------------------------------------------------------  

def con_Pnet(is_diag=False,is_rrvstar=False,**kwargs):
    """Pnet: pseudo-geodesic net: kg/kn=const.
    """
    
def con_pseudogeodesic_pattern(name,is_orient=False,
                               is_aotu=False,
                               is_pq=False,
                               is_dev=False,
                               is_unique_angle=False,
                               coss=None,
                               is_unique_width=False,
                               is_const_width=False,
                               width=None,
                               **kwargs):
    """ based on self.orient_rr_vn=True: X +=[vN,a]
            unit vN://(e1-e3) x (e2-e4); vN*Nv=a^2
            <==> [vN*(e1-e3)=0; vN*(e2-e4)=0; vn^2=1;vN*Nv=a^2]
            
    X=+[oN1; cos1; a,b; width; pn; xy]
        osculating_pln_normal=binormal: on:// (v1-v) x (v3-v)
        <==> on*(v1-v)=0; on*(v3-v)=0; on^2=1; vn * on = cos
      
    "len(cos)==len(a)==numpl; len(on)==len(n)==len(b)==num_rrv4f4"
    
    if orient (orient with vN  & polyline-direction): 
         cos = a^2; on*(vNxt2)=b^2 ####on*(e2-e4)=b^2.
         if aotu: 
             on[ao]*(e2[ao]-e4[ao]) = b[ao]^2
             on[tu]*(e2[tu]-e4[tu]) = b[tu]^2
             
             
    """ 
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nn = kwargs.get('Norient')
    num = mesh.num_rrv4f4
    arr,arr3 = np.arange(num),np.arange(3*num)
    c_n = arr3 + Nn-4*num
    v,v1,v2,v3,v4 =mesh.rrv4f4
    c_v = column3D(v,0,mesh.V) 
    N5 = kwargs.get('N5')
    c_ue1 = column3D(arr,N5-12*num,num)
    c_ue2 = column3D(arr,N5-9*num,num)
    c_ue3 = column3D(arr,N5-6*num,num)
    c_ue4 = column3D(arr,N5-3*num,num)

    def _binormal(X,c_v,c_va,c_vb,c_on):
        "osculating_pln_normal=binormal: on:// (v1-v) x (v3-v)"
        "on*(v1-v)=0; on*(v3-v)=0; on^2=1; no orientation"
        H1,r1 = con_planarity(X,c_va,c_v,c_on)
        H2,r2 = con_planarity(X,c_vb,c_v,c_on)
        H3,r3 = con_unit(X,c_on)
        H = sparse.vstack((H1,H2,H3))
        r = np.r_[r1,r2,r3]
        return H,r
    
    if name=='pseudogeo_1st':
        w = kwargs.get('pseudogeo_1st')
        Nps = kwargs.get('Nps1')
        numpl = mesh.all_rr_polylines_num[0]
        c_va = column3D(v1,0,mesh.V)
        c_vb = column3D(v3,0,mesh.V)
        ind = mesh.all_rr_polylines_vstar_order[0]
    elif name == 'pseudogeo_2nd':
        w = kwargs.get('pseudogeo_2nd')
        Nps = kwargs.get('Nps2')
        numpl = mesh.all_rr_polylines_num[1]
        c_va = column3D(v2,0,mesh.V)
        c_vb = column3D(v4,0,mesh.V)
        ind = mesh.all_rr_polylines_vstar_order[1]

    if is_unique_angle:
        numpl = 1
        c_cos = Nps-numpl
        c_on = arr3 + Nps-3*num - numpl
        Ha,ra = con_constangle2(X,c_on,c_n,c_cos)
        #sin = np.sqrt(1-X[c_cos]**4)*np.ones(num)
        if coss:
            Hu,ru = con_constl(np.array([c_cos],dtype=int),coss,N)
            Ha = sparse.vstack((Ha,Hu))
            ra = np.r_[ra,ru]
            #sin = np.sqrt(1-coss**2)*np.ones(num)
    else:
        "len(cos)==len(a)==numpl; len(on)==len(n)==num_rrv4f4"
        c_cos = np.arange(numpl) + Nps-numpl
        c_on = arr3 + Nps-3*num - numpl
        c_cos_ind = c_cos[ind]
        "multi const.ps-angle for different curves,each crv has a const.angle"
        "vN * oN2 = cos(angle)"
        Ha,ra = con_constangle3(X,c_on,c_n,c_cos_ind) 
        #vcos = np.einsum('ij,ij->i',X[c_on].reshape(-1,3,order='F'),X[c_n].reshape(-1,3,order='F'))
        #sin = np.sqrt(1-vcos**2)##should be np.sqrt(1-X[c_cos_ind]**4)
        
    "binormal:= normal of osculating plane "
    Hon,ron = _binormal(X,c_v,c_va,c_vb,c_on)
    H = sparse.vstack((Hon,Ha))
    r = np.r_[ron,ra]
    #print('on:', np.sum(np.square((Hon*X)-ron)))
    #print('a:', np.sum(np.square((Ha*X)-ra)))
    # print('H:', np.sum(np.square((H*X)-r)))
    
    if is_orient:
        vN = X[c_n].reshape(-1,3,order='F')
        if name=='pseudogeo_1st':
            "vn*on1=cos1=a1^2;on1*(e2-e4)=b1^2"
            t = (X[c_ue1]-X[c_ue3]).reshape(-1,3,order='F')
            c_el,c_er = c_ue2,c_ue4
            c_b = kwargs.get('Nps_orient1')-num+arr
            c_a = kwargs.get('Nps_orient1')-num-numpl+np.arange(numpl)
        elif name == 'pseudogeo_2nd':
            "on2*(e1-e3)=b2^2"
            t = (X[c_ue2]-X[c_ue4]).reshape(-1,3,order='F')
            c_el,c_er = c_ue1,c_ue3
            c_b = kwargs.get('Nps_orient2')-num+arr
            c_a = kwargs.get('Nps_orient2')-num-numpl+np.arange(numpl)
        T = t/np.linalg.norm(t,axis=1)[:,None]
        U = np.cross(vN,T)/np.linalg.norm(np.cross(vN,T),axis=1)[:,None]
        Uxyz = U.flatten('F')
        if is_aotu:
            "everysecond polyline are same orientation"
            if name=='pseudogeo_1st':
                even,odd = mesh.all_rr_polylines_everysecond_index[0]
            elif name == 'pseudogeo_2nd':
                even,odd = mesh.all_rr_polylines_everysecond_index[1]
            pos = np.r_[even,even+num,even+2*num]
            neg = np.r_[odd,odd+num,odd+2*num]

            i_ao = np.r_[np.arange(numpl)[::4],np.arange(numpl)[1::4]]
            i_tu = np.r_[np.arange(numpl)[2::4],np.arange(numpl)[3::4]]
            
            "Ao: orient with vN & t:"
            Hao,rao = con_positive(X,c_cos[i_ao],c_a[i_ao])
            Hpos,rpos = con_orient1(X,c_on[pos],c_el[pos],c_er[pos],c_b[even])
            #Hpos,rpos = con_constangle4(X,c_on[pos],n_xyz[pos],sin[even]) ##need to check
            "Tu: orient with -vN & -t:"
            Htu,rtu = con_negative(X,c_cos[i_tu],c_a[i_tu],len(i_tu))
            Hneg,rneg = con_orient1(X,c_on[neg],c_er[neg],c_el[neg],c_b[odd],True)
            #Hneg,rneg = con_constangle4(X,c_on[neg],n_xyz[neg],sin[odd]) ##need to check
            H = sparse.vstack((H,Hao,Hpos,Htu,Hneg))
            r = np.r_[r,rao,rpos,rtu,rneg]
            # H = sparse.vstack((H,Hao,Hpos))
            # r = np.r_[r,rao,rpos]
            #print('1:', np.sum(np.square((Hao*X)-rao)))
            #print('2:', np.sum(np.square((Hpos*X)-rpos)))
            print('3:', np.sum(np.square((Htu*X)-rtu)))
            print('4:', np.sum(np.square((Hneg*X)-rneg)))
            print('e:', np.sum(np.square((H*X)-r)))
        else:
            "<Curved-pleated structure> case: sharp mountain / valley"
            "orient with vN: vn*on1=cos1=a1^2"
            Ha,ra = con_positive(X,c_cos,c_a)
            #"orient with t: on*(el-er)=b^2"
            #Hb,rb = con_orient1(X, c_on, c_el, c_er, c_b)
            "orient with t: on*(n x t1)=b^2:=sin=sqrt(1-cos^2):=sqrt(1-a^4)"
            Hb,rb = con_orient2(X,c_on,Uxyz,c_b)
            #Hb,rb = con_constangle4(X,c_on,Uxyz,sin)
            H = sparse.vstack((H,Ha,Hb))
            r = np.r_[r,ra,rb]
            #print('cos=a^2:', np.sum(np.square((Ha*X)-ra)))
            #print('on*U=sqrt(1-a^4):', np.sum(np.square((Hb*X)-rb)))
    
    if is_pq: ###No use now. replace by below is_dev
        """ if pq1: the family of 1st-pseudogeodesic rectifying quads are planar
            if pq2: the 2nd.....
            if both: both ... are planar
        """
        id13l,id13r,id24l,id24r = mesh.all_rr_polylines_pq_vstar_order
        onx,ony,onz = c_on[:num],c_on[num:2*num],c_on[2*num:]
        vx,vy,vz = c_v[:num],c_v[num:2*num],c_v[2*num:]
        if name=='pseudogeo_1st':
            "pn1*oni = pn1*onj = pn1*(vi-vj)=0"
            numpq=len(id13l)
            c_pn = kwargs.get('Nps_pq1')-3*numpq+np.arange(3*numpq)
            c_oni = np.r_[onx[id13l],ony[id13l],onz[id13l]]
            c_onj = np.r_[onx[id13r],ony[id13r],onz[id13r]]
            c_vi = np.r_[vx[id13l],vy[id13l],vz[id13l]]
            c_vj = np.r_[vx[id13r],vy[id13r],vz[id13r]]
        elif name == 'pseudogeo_2nd':
            "pn2*oni = pn2*onj = pn2*(vi-vj)=0"
            numpq=len(id24l)
            c_pn = kwargs.get('Nps_pq2')-3*numpq+np.arange(3*numpq)
            c_oni = np.r_[onx[id24l],ony[id24l],onz[id24l]]
            c_onj = np.r_[onx[id24r],ony[id24r],onz[id24r]]
            c_vi = np.r_[vx[id24l],vy[id24l],vz[id24l]]
            c_vj = np.r_[vx[id24r],vy[id24r],vz[id24r]]
        
        def _planar_rectify_srf(c_pn,c_oni,c_onj,c_vi,c_vj):
            "pn*oni = pn*onj = pn*(vi-vj)=0"
            H1,r1 = con_orthogonal_2vectors(X,c_pn,c_oni)
            H2,r2 = con_orthogonal_2vectors(X,c_pn,c_onj)
            H3,r3 = con_planarity(X,c_vi,c_vj,c_pn)
            H4,r4 = con_unit(X,c_pn)
            H = sparse.vstack((H1,H2,H3,H4))
            r = np.r_[r1,r2,r3,r4]
            return H,r
        Hi,ri = _planar_rectify_srf(c_pn,c_oni,c_onj,c_vi,c_vj)
        H = sparse.vstack((H,Hi))
        r = np.r_[r,ri]

    if is_dev:
        """ in fact no need below, since on // eixej,alpha=bet=90
        rectifying strip can be developed into plane. No need PQ.
        Here asking two consecutive edges who share a common ruling vector
        forming two angles alpha + beta = pi
                 / r
          <_____/______>
             ei     ej      r*ei = -r*ej <=> r*(ei+ej)=0
        """   
        if name=='pseudogeo_1st':
            "r*(ei+ej)=0"
            c_ei,c_ej = c_ue1,c_ue3
        elif name == 'pseudogeo_2nd':
            "r*(ei+ej)=0"
            c_ei,c_ej = c_ue2,c_ue4
        Hd,rd = con_ortho(X,c_ei,c_ej,c_on)
        H = sparse.vstack((H,Hd))
        r = np.r_[r,rd]
        ##print('err:', np.sum(np.square((Hd*X)-rd)))

    if is_unique_width or is_const_width:
        "[(A-C)^2; (D-B)^2]=width^2"
        if name=='pseudogeo_1st':
            idA,idC,idD,idB = mesh.all_rr_polylines_aotu1234_index[0]
            Nps_width = kwargs.get('Nps_width1')
            num_strip13 = mesh.all_rr_aotu_strips_num[0]
            num_width = num_strip13[0]+num_strip13[2]
            arr_AC,arr_DB = mesh.all_rr_aotu_strip_width_arr[0]
        elif name == 'pseudogeo_2nd':
            idA,idC,idD,idB = mesh.all_rr_polylines_aotu1234_index[1]
            Nps_width = kwargs.get('Nps_width2')
            num_strip24 = mesh.all_rr_aotu_strips_num[1]
            num_width = num_strip24[0]+num_strip24[2]
            arr_AC,arr_DB = mesh.all_rr_aotu_strip_width_arr[1]
        "NOTE: HAS PROBLEM FOR different len(A)!=len(C); len(D)!=len(B)"
        ia,ib = min(len(idA),len(idC)), min(len(idD),len(idB))
        #print(len(idA),len(idC),len(idD),len(idB),ia,ib)
        idA,idC,idD,idB = idA[:ia],idC[:ia],idD[:ib],idB[:ib]
        #print(len(idA),len(idC),len(idD),len(idB))
        v_AD, v_CB = v[np.r_[idA,idD]], v[np.r_[idC,idB]]
        c_AD = column3D(v_AD,0,mesh.V)
        c_CB = column3D(v_CB,0,mesh.V)
        if is_unique_width:
            c_width = Nps_width-1
            Hw,rw = con_diagonal2(X,c_AD,c_CB,c_width)
        else:
            "c_width ~ [arrAC; arrDB]"
            c_width = Nps_width - num_width + np.arange(num_width)
            c_w = np.array([],dtype=int)
            arr = np.r_[arr_AC, arr_DB]
            for i in range(num_width):
                c_w = np.r_[c_w,np.tile(c_width[i],arr[i])]
            #print(len(v_AD),len(c_w),num_width,len(arr))
            Hw,rw = con_circle(X,c_AD,c_CB,c_w)
        H = sparse.vstack((H,Hw))
        r = np.r_[r,rw]
    
    #print('err:', np.sum(np.square((H*X)-r)))
    return H*w,r*w
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
 