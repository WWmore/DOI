# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:13:42 2025

@author: wanghui
"""
__author__ = 'Hui'
#------------------------------------------------------------------------------
import numpy as np

import itertools

# -----------------------------------------------------------------------------
from geometrylab.optimization.guidedprojectionbase import GuidedProjectionBase

from archgeolab.constraints.constraints_basic import con_planarity_constraints

from archgeolab.constraints.constraints_fairness import con_fairness_4th_different_polylines

from archgeolab.constraints.constraints_net import con_unit_edge,con_orient_rr_vn,\
    con_orthogonal_midline,con_anet,con_snet,con_doi,con_doi__freeform,\
    con_kite,con_pseudogeodesic_pattern
    #,con_cgc,con_pnet

from archgeolab.constraints.constraints_glide import con_glide_in_plane,\
    con_alignment,con_alignments,con_selected_vertices_glide_in_one_plane,\
    con_fix_vertices,con_sharp_corner
                    
from archgeolab.archgeometry.conicSection import interpolate_sphere

from archgeolab.archgeometry.getGeometry import get_strip_from_rulings

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class GP_DOINet(GuidedProjectionBase):
    _N1 = 0
    
    _N2 = 0

    _N3 = 0

    _N4 = 0
    
    _N5 = 0
    

    _mesh = None
    
    _Nanet = 0
    _Nsnet,_Ns_n,_Ns_r = 0,0,0
    _Nsdnet = 0
    
    _Norient = 0
    _Nps1 = 0
    _Nps2 = 0
    _Nps_orient1 = 0
    _Nps_orient2 = 0
    _Nps_width1 = 0
    _Nps_width2 = 0

    def __init__(self):
        GuidedProjectionBase.__init__(self)

        weights = {
            
        'fairness_4diff' :0,
        'fairness_diag_4diff' :0,
 
        'boundary_glide' :0, #Hui in gpbase.py doesn't work, replace here.
        'i_boundary_glide' :0,
        'boundary_z0' :0,
        'sharp_corner' : 0,
        'z0' : 0,

        'unit_edge_vec' : 0,  ## [ei, li]
        'unit_diag_edge_vec' : 0,

        'planarity' : 0,

        'orthogonal' :0,
        
        'DOI' :0,
        
        'Kite' :0,
        
        'CGC' :0,
        'CGC_diagnet' :0,
        'Gnet' : 0,  
        'Gnet_diagnet' : 0, 

        'Anet' : 0,  
        'Anet_diagnet' : 0,  
        
        'Snet' : 0,
        'Snet_diagnet' : 0,
        'Snet_orient' : 1,
        'Snet_constR' : 0,

        'Pnet' :0, ##TODO
        'pseudogeo_1st':0,
        'pseudogeo_2nd':0,

        ##Note: below from geometrylab/optimization/Guidedprojection.py:
        'fixed_vertices' : 1,

        'fixed_corners' : 0,

        'gliding' : 0, # Huinote: glide on boundary, used for itself boundary
        
        }

        self.add_weights(weights)

        self.is_initial = True
        
        self._glide_reference_polyline = None
        self.i_glide_bdry_crv, self.i_glide_bdry_ver = [],[]
        self.assign_coordinates = None
        
        self.orient_rr_vn = False
        
        self.is_GO_or_OG = True
        self.is_diag_or_ctrl = False
        
        self.is_DOI_SIR = False
        self.is_DOI_SIR_diagKite = False
        self.is_Kite_diagGPC = False
        self.is_Kite_diagGPC_SIR = False
        
        ##pseudogeodesic project:
        self.is_psangle1,self.is_psangle2 = False,False
        self.is_pseudogeo_allSameAngle = False
        self.is_pseudogeo_limitAngle = 0
        self.pseudogeo_1st_constangle = None
        self.pseudogeo_2nd_constangle = None
        self.is_pseudogeo_uniquewidth=self.is_pseudogeo_constwidth = False
        self.data_pseudogeodesic_binormal = None #=[an,oN1,oN2,cos13,cos24]

        self.if_uniqradius = False
        self.assigned_snet_radius = 0

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        self.initialization()

    @property
    def max_weight(self):    
        return max(
                   self.get_weight('boundary_glide'),
                   self.get_weight('planarity'),
                   
                   self.get_weight('unit_edge_vec'),
                   self.get_weight('unit_diag_edge_vec'),
                   
                   self.get_weight('orthogonal'),
 
                   self.get_weight('DOI'),
                   
                   self.get_weight('Kite'),
                   
                   self.get_weight('CGC'),
                   self.get_weight('CGC_diagnet'),
                   self.get_weight('Pnet'),
                   
                   self.get_weight('Anet'),
                   self.get_weight('Anet_diagnet'),
                   
                   self.get_weight('Snet'),
                   self.get_weight('Snet_diagnet'),

                   1)
    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self,angle):
        if angle != self._angle:
            self.mesh.angle=angle
        self._angle = angle
        
    @property
    def glide_reference_polyline(self):
        if self._glide_reference_polyline is None:
            #polylines = self.mesh.boundary_polylines() #Hui note: problem for boundary_polylines()
            polylines = self.mesh.boundary_curves(corner_split=False)[0]   
            ##print(polylines)
            
            N = 5
            try:
                for polyline in polylines:
                    polyline.refine(N)
            except:
                ###Add below for closed 1 boundary of reference mesh
                from geometrylab.geometry import Polyline
                polyline = Polyline(self.mesh.vertices[polylines],closed=True)  
                polyline.refine(N)

            self._glide_reference_polyline = polyline
        return self._glide_reference_polyline

    @glide_reference_polyline.setter
    def glide_reference_polyline(self,polyline):
        self._glide_reference_polyline = polyline        

    #--------------------------------------------------------------------------
    #                               Initialization
    #--------------------------------------------------------------------------

    def set_weights(self):
        self.set_weight('fixed_vertices', 10 * self.max_weight)
        self.set_weight('gliding', 10 * self.max_weight)
        if self.get_weight('fixed_corners') != 0:
            self.set_weight('fixed_corners', 10 * self.max_weight)
        
        if self.get_weight('pseudogeo_1st') or self.get_weight('pseudogeo_2nd'):
            self.orient_rr_vn = True
 
        if self.orient_rr_vn:
            self.set_weight('unit_edge_vec', 1)    

          
    def set_dimensions(self): # Huinote: be used in guidedprojectionbase
        "X:= [Vx,Vy,Vz]"
        V = self.mesh.V
        F = self.mesh.F
        N = 3*V
        N1 = N2 = N3 = N4 = N5 = N
        num_rrstar = self.mesh.num_rrv4f4 ##may have problem for only one strip
        
        Norient = N

        Nanet = N
        Nsnet = Ns_n = Ns_r = N
        
        Nps1 = Nps2 = Nps_orient1 = Nps_orient2 = N
        Nps_width1 = Nps_width2 = N
        #---------------------------------------------
        if self.get_weight('planarity') != 0:
            "X += [Nx,Ny,Nz]"
            N += 3*F
            N1 = N2 = N3 = N4 = N

        if self.get_weight('unit_edge_vec'): #Gnet, AGnet
            "X+=[le1,le2,le3,le4,ue1,ue2,ue3,ue4]"
            "for Anet, AGnet, DGPC"
            N += 16*self.mesh.num_rrv4f4
            N5 = N
        elif self.get_weight('unit_diag_edge_vec'): #Gnet_diagnet
            "le1,le2,le3,le4,ue1,ue2,ue3,ue4 "
            N += 16*self.mesh.num_rrv4f4
            N5 = N
        
        if self.orient_rr_vn:
            "X+=[vn, a], vN * Nv = a^2>=0; Nv is given orient-vertex-normal"
            N += 4*num_rrstar
            Norient = N

        if self.get_weight('Anet') or self.get_weight('Anet_diagnet'):
            N += 3*self.mesh.num_rrv4f4#3*num_regular
            Nanet = N

            
        ### Snet(_diag) project:
        if self.get_weight('Snet') or self.get_weight('Snet_diagnet'):
            num_snet = self.mesh.num_rrv4f4 
            N += 11*num_snet  
            Nsnet = N
            if self.get_weight('Snet_orient'):
                N +=4*num_snet  
                Ns_n = N
            if self.get_weight('Snet_constR'):
                N += 1
                Ns_r = N

        if self.get_weight('pseudogeo_1st') or self.get_weight('pseudogeo_2nd'):  
            "based on orient_rr_vn=True"
            num13,num24 = self.mesh.all_rr_polylines_num
            if self.get_weight('pseudogeo_1st'):
                "X +=[on1,cos] "
                N += 3*num_rrstar
                if self.is_pseudogeo_allSameAngle: ##unique angle
                    "unique const.ps-angle, one variable"
                    num_cos=1
                else:
                    "multi const.ps-angle for different curves"
                    num_cos=num13
                N += num_cos 
                Nps1 = N
                
                if self.is_pseudogeo_orient:
                    N += num_cos + num_rrstar
                    Nps_orient1 = N
                    
                num_width=0
                if self.is_pseudogeo_uniquewidth:
                    num_width=1
                elif self.is_pseudogeo_constwidth:
                    "support-structure-strip for AC & DB"
                    if False:
                        if int(num13%4)==2 or int(num13%4)==3:
                            num_width=int(num13/4) + 1 + int((num13-2)/4)
                        else:
                            num_width=int(num13/4) + int((num13-2)/4)
                    else:
                        "below should equal to the above"
                        num_strip13 = self.mesh.all_rr_aotu_strips_num[0]
                        num_width = num_strip13[0]+num_strip13[2]
                N += num_width 
                Nps_width1 = N   
                    
            if self.get_weight('pseudogeo_2nd'):
                "X +=[on2,cos] "
                N += 3*num_rrstar
                if self.is_pseudogeo_allSameAngle: ##unique angle
                    "unique const.ps-angle, one variable"
                    num_cos=1
                else:
                    "multi const.ps-angle for different curves"
                    num_cos=num24
                N += num_cos
                Nps2 = N
                
                if self.is_pseudogeo_orient:
                    N += num_cos + num_rrstar
                    Nps_orient2 = N

                num_width=0
                if self.is_pseudogeo_uniquewidth:
                    num_width=1
                elif self.is_pseudogeo_constwidth:
                    "support-structure-strip for AC & DB"
                    if False:
                        if int(num24%4)==2 or int(num24%4)==3:
                            num_width=int(num24/4) + 1 + int((num24-2)/4)
                        else:
                            num_width=int(num24/4) + int((num24-2)/4)
                    else:
                        "below should equal to the above"
                        num_strip24 = self.mesh.all_rr_aotu_strips_num[1]
                        num_width = num_strip24[0]+num_strip24[2]
                N += num_width 
                Nps_width2 = N  
        #---------------------------------------------
        if N1 != self._N1 or N2 != self._N2:
            self.reinitialize = True
        if N3 != self._N3 or N4 != self._N4:
            self.reinitialize = True
        if self._N2 - self._N1 == 0 and N2 - N1 > 0:
            self.mesh.reinitialize_densities()

        if N5 != self._N5:
            self.reinitialize = True
        if Nanet != self._Nanet:
            self.reinitialize = True
        if Nsnet != self._Nsnet:
            self.reinitialize = True
        if Ns_n != self._Ns_n:
            self.reinitialize = True
        if Ns_r != self._Ns_r:
            self.reinitialize = True
            
        if Norient != self._Norient:
            self.reinitialize = True    
        if Nps1 != self._Nps1:
            self.reinitialize = True  
        if Nps2 != self._Nps2:
            self.reinitialize = True 
        if Nps_orient1 != self._Nps_orient1:
            self.reinitialize = True  
        if Nps_orient2 != self._Nps_orient2:
            self.reinitialize = True  
        if Nps_width1 != self._Nps_width1:
            self.reinitialize = True  
        if Nps_width2 != self._Nps_width2:
            self.reinitialize = True  
        #----------------------------------------------
        self._N = N
        self._N1 = N1
        self._N2 = N2
        self._N3 = N3
        self._N4 = N4
        self._N5 = N5
        self._Nanet = Nanet
        self._Nsnet,self._Ns_n,self._Ns_r = Nsnet,Ns_n,Ns_r

        self._Norient = Norient
        self._Nps1 = Nps1
        self._Nps2 = Nps2
        self._Nps_orient1 = Nps_orient1
        self._Nps_orient2 = Nps_orient2
        self._Nps_width1 = Nps_width1
        self._Nps_width2 = Nps_width2
        
        self.build_added_weight() # Hui add
        
        
    def initialize_unknowns_vector(self):
        "X:= [Vx,Vy,Vz]"
        X = self.mesh.vertices.flatten('F')
        if self.get_weight('planarity') != 0:
            "X += [Nx,Ny,Nz]; len=3F"
            normals = self.mesh.face_normals()
            X = np.hstack((X, normals.flatten('F')))

        if self.get_weight('unit_edge_vec'):
            _,l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_unit_edge(rregular=True)
            X = np.r_[X,l1,l2,l3,l4]
            X = np.r_[X,E1.flatten('F'),E2.flatten('F'),E3.flatten('F'),E4.flatten('F')]

        elif self.get_weight('unit_diag_edge_vec'):
            _,l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_diag_unit_edge()
            X = np.r_[X,l1,l2,l3,l4]
            X = np.r_[X,E1.flatten('F'),E2.flatten('F'),E3.flatten('F'),E4.flatten('F')]

        if self.orient_rr_vn:
            _,vN,a = self.mesh.get_v4_orient_unit_normal()
            X = np.r_[X,vN.flatten('F'),a]

        if self.get_weight('Anet') or self.get_weight('Anet_diagnet'):
            if self.get_weight('Anet'):
                if True:
                    "only r-regular vertex"
                    v = self.mesh.ver_rrv4f4
                else:
                    v = self.mesh.ver_regular
            elif self.get_weight('Anet_diagnet'):
                v = self.mesh.rr_star_corner[0]
            V4N = self.mesh.vertex_normals()[v]
            X = np.r_[X,V4N.flatten('F')]

        ### CNC:
        if self.get_weight('Snet') or self.get_weight('Snet_diagnet'):
            r = self.get_weight('Snet_constR')
            is_diag = False if self.get_weight('Snet') else True
            x_snet,Nv4 = self.get_snet(r,is_diag)
            X = np.r_[X,x_snet]

        ### Pnet:
        if self.get_weight('pseudogeo_1st') or self.get_weight('pseudogeo_2nd'):
            """ based on orient_rr_vs: X +=[vN,a] (vN*Nv=a^2)
            X=+[oN1; cos1; a,b; width; pn; xy]
            if AoTu_structure: oN1,oN2 is oriented with up/down direction;
                               oN only appears in one continuous quadrant;
                               oriented with orientT=True, no relation with vN
            else: 
                if orientT(default): oN1,oN2 are oriented with t1,t2 = e1-e3, e2-e4
                if orientN: oN is oriented with the oriented-vertex-normal
            """
            if self.data_pseudogeodesic_binormal is None:
                "Pl. Click &  Check binormals first: "
                _,oN1,oN2,cs13,cs24 = self.pseudogeodesic_binormal()
            else:
                _,oN1,oN2,cs13,cs24 = self.data_pseudogeodesic_binormal
            cos1,cos2 = np.abs(cs13[0]), np.abs(cs24[0])
            #_,_,_,_,_,e1,e2,e3,e4 = self.mesh.get_v4_unit_edge(True)
            _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents()
            vN = self.mesh.get_v4_unit_normal()[2] ##vN:=ut1xut2/||
            
            if self.get_weight('pseudogeo_1st'):
                if self.is_pseudogeo_allSameAngle:
                    cos1 = np.mean(cos1)
                X = np.r_[X,oN1.flatten('F'), cos1]
                
                if self.is_pseudogeo_orient:
                    "vn*on1=cos1=a1^2;on1*(vNxut1)=sin1=b1^2; a1^4+b1^4==1"
                    a = np.sqrt(np.abs(cos1))
                    U = np.cross(vN,ut1)/np.linalg.norm(np.cross(vN,ut1),axis=1)[:,None]
                    b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN1,U)))
                    #"vn*on1=cos1=a1^2;on1*(e2-e4)=b1^2"
                    #b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN1,e2-e4)))
                    X = np.r_[X,a,b]                
                
                if self.is_pseudogeo_uniquewidth:
                    width = self.get_pseudogeodesic_constwidth(unique=True)
                    X = np.r_[X,width]
                elif self.is_pseudogeo_constwidth:
                    width = self.get_pseudogeodesic_constwidth()
                    X = np.r_[X,width]


            if self.get_weight('pseudogeo_2nd'):
                if self.is_pseudogeo_allSameAngle:
                    cos2 = np.mean(cos2)
                X = np.r_[X,oN2.flatten('F'), cos2]
                
                if self.is_pseudogeo_orient:
                    "either or both family(ies) have oriented binormals between <vn,t2(t1)>"
                    "vn*on1=cos1=a1^2;on1*(vNxut1)=sin1=b1^2; a1^4+b1^4==1"
                    a = np.sqrt(np.abs(cos2))
                    U = np.cross(vN,ut2)/np.linalg.norm(np.cross(vN,ut2),axis=1)[:,None]
                    b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN2,U)))
                    #"vn*on2=cos2=a2^2;on2*(e1-e3)=b2^2"
                    #b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN2,e1-e3)))
                    X = np.r_[X,a,b]
                    
                if self.is_pseudogeo_uniquewidth:
                    width = self.get_pseudogeodesic_constwidth(another=True,unique=True)
                    X = np.r_[X,width]
                elif self.is_pseudogeo_constwidth:
                    width = self.get_pseudogeodesic_constwidth(another=True)
                    X = np.r_[X,width]       
        #-----------------------
        
        self._X = X
        self._X0 = np.copy(X)
            
        self.build_added_weight() # Hui add

    #--------------------------------------------------------------------------
    #                       Getting (initilization + Plotting):
    #--------------------------------------------------------------------------

    def get_snet(self,is_r,is_diag=False,is_orient=True):
        """
        each vertex has one [a,b,c,d,e] for sphere equation:
            f=a(x^2+y^2+z^2)+(bx+cy+dz)+e=0
            when a=0; plane equation
            (x-x0)^2+(y-y0)^2+(z-z0)^2=R^2
            M = (x0,y0,z0)=-(b,c,d)/2a
            R^2 = (b^2+c^2+d^2-4ae)/4a^2
            unit_sphere_normal==-(2*A*Vx+B, 2*A*Vy+C, 2*A*Vz+D), 
            (note direction: from vertex to center)
            
        X += [V^2,A,B,C,D,E,a_sqrt]
        if orient:
            X += [n4,n4_sqrt], n4=-[2ax+b,2ay+c,2az+d]
        if r:
            X += [r]
        if angle
           X += [l1,l2,l3,l4,ue1,ue2,ue3,ue4]
        """
        V = self.mesh.vertices
        if is_diag:
            s0,s1,s2,s3,s4 = self.mesh.rr_star_corner
        else:
            s0,s1,s2,s3,s4 = self.mesh.rrv4f4
        S0,S1,S2,S3,S4 = V[s0],V[s1],V[s2],V[s3],V[s4]
        centers,radius,coeff,Nv4 = interpolate_sphere(S0,S1,S2,S3,S4)
        VV = np.linalg.norm(np.vstack((S0,S1,S2,S3,S4)),axis=1)**2
        A,B,C,D,E = coeff.reshape(5,-1)
        A_sqrt = np.sqrt(A)
        XA = np.r_[VV,A,B,C,D,E,A_sqrt]
        if is_orient: ##always True
            B,C,D,Nv4,n4_sqrt = self.mesh.orient(S0,A,B,C,D,Nv4)
            XA = np.r_[VV,A,B,C,D,E,A_sqrt]  
            XA = np.r_[XA, Nv4.flatten('F'),n4_sqrt]
        if is_r:
            r = np.mean(radius)
            XA = np.r_[XA,r]
        return XA, Nv4

    def get_snet_data(self,is_diag=False, ##note: combine together suitable for diagonal
                      center=False,normal=False,tangent=False,ss=False,
                      is_diag_binormal=False):
        "at star = self.rr_star"
        V = self.mesh.vertices
        if is_diag:
            s0,s1,s2,s3,s4 = self.mesh.rr_star_corner
        else:
            s0,s1,s2,s3,s4 = self.mesh.rrv4f4
            
        S0,S1,S2,S3,S4 = V[s0],V[s1],V[s2],V[s3],V[s4]
        centers,r,coeff,Nv4 = interpolate_sphere(S0,S1,S2,S3,S4)
        if self.get_weight('Snet_orient'):
            A,B,C,D,E = coeff.reshape(5,-1)
            _,_,_,Nv4,_ = self.mesh.orient(S0,A,B,C,D,Nv4)
            #Nv4 = self.mesh.get_v4_orient_unit_normal()[1][self.mesh.ind_rr_star_v4f4]
            centers = S0+r[:,None]*Nv4
        if center:
            er0 = np.abs(np.linalg.norm(S0-centers,axis=1)-r)
            er1 = np.abs(np.linalg.norm(S1-centers,axis=1)-r)
            er2 = np.abs(np.linalg.norm(S2-centers,axis=1)-r)
            er3 = np.abs(np.linalg.norm(S3-centers,axis=1)-r)
            er4 = np.abs(np.linalg.norm(S4-centers,axis=1)-r)
            #err = (er0+er1+er2+er3+er4) / 5
            err = np.sqrt(er0**2+er1**2+er2**2+er3**2+er4**2)/r
            print('radii:[min,mean,max]=','%.3f'%np.min(r),'%.3f'%np.mean(r),'%.3f'%np.max(r))
            print('Err:[min,mean,max]=','%.3g'%np.min(err),'%.3g'%np.mean(err),'%.3g'%np.max(err))
            return centers,err
        elif normal:
            # n = np.cross(C3-C1,C4-C2)
            #n = S0-centers
            n = Nv4
            un = n / np.linalg.norm(n,axis=1)[:,None]
            return S0,un
        elif is_diag_binormal:
            "only work for SSG/GSS/SSGG/GGSS-project, not for general Snet"
            n = Nv4
            un = n / np.linalg.norm(n,axis=1)[:,None]
            if is_diag:
                _,sa,sb,sc,sd = self.mesh.rrv4f4
            else:
                _,sa,sb,sc,sd = self.mesh.rr_star_corner
            t1 = (V[sa]-V[sc])/np.linalg.norm(V[sa]-V[sc], axis=1)[:,None]
            t2 = (V[sb]-V[sd])/np.linalg.norm(V[sb]-V[sd], axis=1)[:,None]
            "note works for SSG..case, since un,sa-v,sc are coplanar"
            bin1 = np.cross(un, t1)
            bin2 = np.cross(un, t2)
            bin1 = bin1 / np.linalg.norm(bin1, axis=1)[:,None]
            bin2 = bin2 / np.linalg.norm(bin2, axis=1)[:,None]
            return S0, bin1, bin2
        elif tangent:
            inn,_ = self.mesh.get_rr_vs_bounary()
            V0,V1,V2,V3,V4 = S0[inn],S1[inn],S2[inn],S3[inn],S4[inn]
            l1 = np.linalg.norm(V1-V0,axis=1)
            l2 = np.linalg.norm(V2-V0,axis=1)
            l3 = np.linalg.norm(V3-V0,axis=1)
            l4 = np.linalg.norm(V4-V0,axis=1)
            t1 = (V1-V0)*(l3**2)[:,None] - (V3-V0)*(l1**2)[:,None]
            t1 = t1 / np.linalg.norm(t1,axis=1)[:,None]
            t2 = (V2-V0)*(l4**2)[:,None] - (V4-V0)*(l2**2)[:,None]
            t2 = t2 / np.linalg.norm(t2,axis=1)[:,None]
            return V0,t1,t2
        elif ss:
            n = Nv4
            un = n / np.linalg.norm(n,axis=1)[:,None]
            return s0,un   
        return S0,S1,S2,S3,S4,centers,r,coeff,Nv4
  
    # -------------------------------------------------------------------------
    #                                 Build
    # -------------------------------------------------------------------------

    def build_iterative_constraints(self):
        self.build_added_weight() # Hui change
        
        H, r = self.mesh.iterative_constraints(**self.weights) ##NOTE: in gridshell.py
        self.add_iterative_constraint(H, r, 'mesh_iterative')
        
        H, r = self.mesh.fairness_energy(**self.weights) ##NOTE: in gridshell.py
        self.add_iterative_constraint(H, r, 'fairness')
        
        if self.get_weight('fairness_4diff'):
            pl1,pl2 = self.mesh.all_rr_continuous_polylist
            pl1.extend(pl2)
            H,r = con_fairness_4th_different_polylines(pl1,**self.weights)
            self.add_iterative_constraint(H, r, 'fairness_4diff') 
        if self.get_weight('fairness_diag_4diff'):
            pl1 = self.mesh.all_rr_diag_polylist[0][0]
            pl2 = self.mesh.all_rr_diag_polylist[1][0]
            pl1.extend(pl2)
            H,r = con_fairness_4th_different_polylines(pl1,diag=True,**self.weights)
            self.add_iterative_constraint(H, r, 'fairness_diag_4diff')     
        
        if self.get_weight('planarity'):
            #"use Hui's way not Davide's"
            H,r = con_planarity_constraints(**self.weights)  
            self.add_iterative_constraint(H, r, 'planarity')

            
        ###-------partially shared-used codes:---------------------------------
        if self.get_weight('unit_edge_vec'): 
            H,r = con_unit_edge(rregular=True,**self.weights)
            self.add_iterative_constraint(H, r, 'unit_edge')
        elif self.get_weight('unit_diag_edge_vec'): 
            H,r = con_unit_edge(rregular=True,**self.weights)
            self.add_iterative_constraint(H, r, 'unit_diag_edge_vec')

        if self.orient_rr_vn:
            "rr_vn orients samely as Nv"
            H,r = con_orient_rr_vn(**self.weights) ##weight=1
            self.add_iterative_constraint(H, r, 'orient_vn')

        if self.get_weight('boundary_z0') !=0:
            z = 0
            v = np.array([816,792,768,744,720,696,672,648,624,600,576,552,528,504,480,456,432,408,384,360,336,312,288,264,240,216,192,168,144,120,96,72,48,24,0],dtype=int)
            H,r = con_selected_vertices_glide_in_one_plane(v,2,z,**self.weights)              
            self.add_iterative_constraint(H, r, 'boundary_z0')
            
        if self.assign_coordinates is not None:
            index,Vf = self.assign_coordinates
            H,r = con_fix_vertices(index, Vf.flatten('F'),**self.weights)
            self.add_iterative_constraint(H, r, 'fix_pts')

        if self.get_weight('boundary_glide'):
            "the whole boundary"
            refPoly = self.glide_reference_polyline
            glideInd = self.mesh.boundary_curves(corner_split=False)[0] 
            w = self.get_weight('boundary_glide')
            H,r = con_alignment(w, refPoly, glideInd,**self.weights)
            self.add_iterative_constraint(H, r, 'boundary_glide')
        elif self.get_weight('i_boundary_glide'):
            "the i-th boundary"
            refPoly = self.i_glide_bdry_crv
            glideInd = self.i_glide_bdry_ver
            if len(glideInd)!=0:
                w = self.get_weight('i_boundary_glide')
                H,r = con_alignments(w, refPoly, glideInd,**self.weights)
                self.add_iterative_constraint(H, r, 'iboundary_glide')

        if self.get_weight('sharp_corner'):
            H,r = con_sharp_corner(move=0,**self.weights)
            self.add_iterative_constraint(H,r, 'sharp_corner')
            
        
        if self.get_weight('z0') !=0:
            H,r = con_glide_in_plane(2,**self.weights)
            self.add_iterative_constraint(H,r, 'z0')     
            
        ###------- net construction: ------------------------------------------
            
        if self.get_weight('orthogonal'):
            H,r = con_orthogonal_midline(**self.weights)
            self.add_iterative_constraint(H, r, 'orthogonal')

        
        if self.get_weight('DOI'):
            yes1, yes2 = self.is_DOI_SIR, self.is_DOI_SIR_diagKite
            if True:
                H,r = con_doi__freeform(self.is_GO_or_OG,yes1,yes2,**self.weights)
            else:
                "works well, but only for patch or rotational mesh"
                H,r = con_doi(self.is_GO_or_OG,yes1,**self.weights)
            self.add_iterative_constraint(H, r, 'DOI')
            
        if self.get_weight('Kite'):
            yes1,yes2 = self.is_Kite_diagGPC, self.is_Kite_diagGPC_SIR
            H,r = con_kite(self.is_GO_or_OG,yes1,yes2,**self.weights)
            self.add_iterative_constraint(H, r, 'Kite')
        #elif self.get_weight('Kite_diagnet'): #Hui: works but not use in alignnet on SIR-net
            #H,r = con_kite_diagnet(self.is_GO_or_OG,**self.weights)
            #self.add_iterative_constraint(H, r, 'Kite_diagnet')
            
            
        if self.get_weight('Anet'):
            H,r = con_anet(rregular=True,**self.weights)
            self.add_iterative_constraint(H, r, 'Anet')
        elif self.get_weight('Anet_diagnet'):
            H,r = con_anet(rregular=True,is_diagnet=True,**self.weights)
            self.add_iterative_constraint(H, r, 'Anet_diagnet')

        if self.get_weight('Snet'):
            orientrn = self.mesh.new_vertex_normals()
            H,r = con_snet(orientrn,
                           is_uniqR=self.if_uniqradius,
                           assigned_r=self.assigned_snet_radius,
                           **self.weights)
            self.add_iterative_constraint(H, r, 'Snet') 
        if self.get_weight('Snet_diagnet'):
            orientrn = self.mesh.new_vertex_normals()
            H,r = con_snet(orientrn,is_diagnet=True,
                           is_uniqR=self.if_uniqradius,
                           assigned_r=self.assigned_snet_radius,
                           **self.weights)
            self.add_iterative_constraint(H, r, 'Snet_diag') 

        if self.get_weight('pseudogeo_1st') or self.get_weight('pseudogeo_2nd'):
            "based on self.orient_rr_vn = True"
            is_unique_angle, coss = False, None
            is_unique_width = self.is_pseudogeo_uniquewidth
            is_const_width, width = self.is_pseudogeo_constwidth, None
            is_pq = False #self.is_pseudogeo_rectify_pq ##TODO:hui remove
            is_dev = False #self.is_pseudogeo_rectify_dvlp ##TODO:hui remove
            is_orient = self.is_pseudogeo_orient
            is_aotu = False #self.is_pseudogeo_aotu
            if self.get_weight('pseudogeo_1st'):
                "along each curve, angle is const. but different from each other"
                name = 'pseudogeo_1st'
                if self.is_pseudogeo_allSameAngle:
                    "there is uniqe const. angle"
                    is_unique_angle=True
                    coss = np.cos(self.pseudogeo_1st_constangle/180.0*np.pi)
                H,r = con_pseudogeodesic_pattern(name,is_orient,
                                                 is_aotu,
                                                 is_pq,
                                                 is_dev,
                                                 is_unique_angle,
                                                 coss,
                                                 is_unique_width,
                                                 is_const_width,
                                                 width,
                                                 **self.weights)
                self.add_iterative_constraint(H, r, 'pseudogeo_1st')
            if self.get_weight('pseudogeo_2nd'):
                name = 'pseudogeo_2nd'
                if self.is_pseudogeo_allSameAngle:
                    "there is uniqe const. angle"
                    is_unique_angle=True
                    coss = np.cos(self.pseudogeo_2nd_constangle/180.0*np.pi)
                H,r = con_pseudogeodesic_pattern(name,
                                                 is_orient,
                                                 is_aotu,
                                                 is_pq,
                                                 is_dev,
                                                 is_unique_angle,
                                                 coss,
                                                 is_unique_width,
                                                 is_const_width,
                                                 width,
                                                 **self.weights)
                self.add_iterative_constraint(H, r, 'pseudogeo_2nd')
                
        ###--------------------------------------------------------------------                
   
        self.is_initial = False   
            
        #print('-'*10)
        print(' Err_total: = ','%.3e' % np.sum(np.square(self._H*self.X-self._r)))
        #print('-'*10)
        
    def build_added_weight(self): # Hui add
        self.add_weight('mesh', self.mesh)
        self.add_weight('N', self.N)
        self.add_weight('X', self.X)
        self.add_weight('N1', self._N1)
        self.add_weight('N2', self._N2)
        self.add_weight('N3', self._N3)
        self.add_weight('N4', self._N4)
        self.add_weight('N5', self._N5)
        self.add_weight('Nanet', self._Nanet)
        self.add_weight('Nsnet', self._Nsnet)
        self.add_weight('Ns_n', self._Ns_n)
        self.add_weight('Ns_r', self._Ns_r)
        
        self.add_weight('Norient', self._Norient)
        self.add_weight('Nps1', self._Nps1)
        self.add_weight('Nps2', self._Nps2)
        self.add_weight('Nps_orient1', self._Nps_orient1)
        self.add_weight('Nps_orient2', self._Nps_orient2)
        self.add_weight('Nps_width1', self._Nps_width1)
        self.add_weight('Nps_width2', self._Nps_width2)

    def values_from_each_iteration(self,**kwargs):
        if kwargs.get('unit_edge_vec'):
            _,l1,l2,l3,l4,_,_,_,_ = self.mesh.get_v4_unit_edge(rregular=True)
            Xi = np.r_[l1,l2,l3,l4]
            return Xi

        if kwargs.get('unit_diag_edge_vec'):
            _,l1,l2,l3,l4,_,_,_,_ = self.mesh.get_v4_diag_unit_edge()
            Xi = np.r_[l1,l2,l3,l4]
            return Xi

    def build_constant_constraints(self): #copy from guidedprojection,need to check if it works 
        self.add_weight('N', self.N)
        H, r = self.mesh.constant_constraints(**self.weights)
        self.add_constant_constraint(H, r, 'mesh_constant')

    def build_constant_fairness(self): #copy from guidedprojection,need to check if it works 
        self.add_weight('N', self.N)
        K, s = self.mesh.fairness_energy(**self.weights)
        self.add_constant_fairness(K, s)
  
    def post_iteration_update(self): #copy from guidedprojection,need to check if it works 
        V = self.mesh.V
        self.mesh.vertices[:,0] = self.X[0:V]
        self.mesh.vertices[:,1] = self.X[V:2*V]
        self.mesh.vertices[:,2] = self.X[2*V:3*V]

    def on_reinitilize(self): #copy from guidedprojection,need to check if it works 
        self.mesh.reinitialize_force_densities()
    #--------------------------------------------------------------------------
    #                                  Results
    #--------------------------------------------------------------------------

    def vertices(self):
        V = self.mesh.V
        vertices = self.X[0:3*V]
        vertices = np.reshape(vertices, (V,3), order='F')
        return vertices

    def face_normals(self, initialized=False):
        if self.get_weight('planarity') == 0:
            return None
        if initialized:
            X = self._X0
        else:
            X = self.X
        V = self.mesh.V
        F = self.mesh.F
        normals = X[3*V:3*V+3*F]
        normals = np.reshape(normals, (F,3), order='F')
        return normals


    #------------------------------------------------------------------------
    
    def get_orient_rr_normal(self,is_diag=False,initialized=True):
        if initialized or self.is_initial or not self.orient_rr_vn:
            return self.mesh.get_v4_orient_unit_normal(is_diag) ##==[an,vN,a]
        elif self.orient_rr_vn:
            v = self.mesh.ver_rrv4f4
            an = self.mesh.vertices[v]
            num = self.mesh.num_rrv4f4
            c_n = np.arange(3*num) + self._Norient-4*num 
            vN = self.X[c_n].reshape(-1,3,order='F')
            return [an,vN,0]

    def pseudogeodesic_binormal(self, is_initial=False, GO_Ps=False,
                                      is_orientT=False,is_orientN=False,
                                      is_remedy=False, is_smooth=False):
        "save as self.data_pseudogeodesic_binormal"
        V = self.mesh.vertices
        v = self.mesh.ver_rrv4f4   
        an = V[v]
        if self.is_initial or is_initial: 
            v,v1,v2,v3,v4 = self.mesh.rrv4f4  
            vN = self.mesh.get_v4_orient_unit_normal()[1]
            _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents()
            T2,T1 = np.cross(vN,ut1), np.cross(vN,ut2)
            oN1 = np.cross(V[v1]-V[v],V[v3]-V[v])
            oN2 = np.cross(V[v2]-V[v],V[v4]-V[v])
            if is_remedy:
                lon1 = np.linalg.norm(oN1, axis=1)
                i = np.where(lon1<1e-4)[0]
                for ii in i:
                    j,k = np.where(v==v1[ii])[0],np.where(v==v3[ii])[0]
                    if len(j)!=0 and len(k)!=0:
                        oN1[ii] = (oN1[j]+oN1[k])/np.linalg.norm(oN1[j]+oN1[k])
                    elif len(j)!=0:
                        oN1[ii] = oN1[j]/np.linalg.norm(oN1[j])
                    elif len(k)!=0:
                        oN1[ii] = oN1[k]/np.linalg.norm(oN1[k])
                    else:
                        oN1[ii] = vN[ii]
                        
                lon2 = np.linalg.norm(oN2, axis=1)
                i = np.where(lon2<1e-4)[0]
                for ii in i:
                    j,k = np.where(v==v2[ii])[0],np.where(v==v4[ii])[0]
                    if len(j)!=0 and len(k)!=0:
                        oN2[ii] = (oN2[j]+oN2[k])/np.linalg.norm(oN2[j]+oN2[k])
                    elif len(j)!=0:
                        oN2[ii] = oN2[j]/np.linalg.norm(oN2[j])
                    elif len(k)!=0:
                        oN2[ii] = oN2[k]/np.linalg.norm(oN2[k])
                    else:
                        oN2[ii] = vN[ii]

            oN1 = oN1/np.linalg.norm(oN1, axis=1)[:,None]
            oN2 = oN2/np.linalg.norm(oN2, axis=1)[:,None]
            
            if is_orientT or is_orientN:
                i1 = np.where(np.einsum('ij,ij->i',T2,oN1)<0)[0]
                i2 = np.where(np.einsum('ij,ij->i',T1,oN2)<0)[0] ##note sign

                j1 = np.where(np.einsum('ij,ij->i',vN,oN1)<0)[0]
                j2 = np.where(np.einsum('ij,ij->i',vN,oN2)<0)[0]

                if is_orientT and not is_orientN:
                    oN1[i1] = -oN1[i1] 
                    oN2[i2] = -oN2[i2] 
                        
                elif is_orientN and not is_orientT: ##note: suitable for crv_pleat but not AoTu
                    oN1[j1] = -oN1[j1]  
                    oN2[j2] = -oN2[j2] 
                    
                elif is_orientT and is_orientN:
                    "orientT:[a,b], orientN:[b,c]==> [a',b',c']=[2*N-a,-b,2*t-c]"
                    a1 = np.setdiff1d(i1,j1)
                    c1 = np.setdiff1d(j1,i1)
                    b1 = np.intersect1d(i1,j1)
                    oN1[a1] = 2*vN[a1]-oN1[a1]
                    oN1[b1] = -oN1[b1]
                    oN1[c1] = 2*T2[c1]-oN1[c1]
                    
                    a2 = np.setdiff1d(i2,j2)
                    c2 = np.setdiff1d(j2,i2)
                    b2 = np.intersect1d(i2,j2)
                    oN2[a2] = 2*vN[a2]-oN2[a2]
                    oN2[b2] = -oN2[b2]
                    oN2[c2] = 2*T1[c2]-oN2[c2]
                    
                oN1 = oN1/np.linalg.norm(oN1, axis=1)[:,None]
                oN2 = oN2/np.linalg.norm(oN2, axis=1)[:,None]
                
            if is_smooth:
                from huilab.huimesh.smooth import fair_vectors
                v13l,v13c,v13r = [],[],[]
                v24l,v24c,v24r = [],[],[]
                for i in range(len(v)):
                    if v1[i] in v and v3[i] in v:
                        v13c.append(i)
                        v13l.append(np.where(v==v1[i])[0][0])
                        v13r.append(np.where(v==v3[i])[0][0])
                    if v2[i] in v and v4[i] in v:
                        v24c.append(i)
                        v24l.append(np.where(v==v2[i])[0][0])
                        v24r.append(np.where(v==v4[i])[0][0])
                oN1 = fair_vectors(oN1,v13l,v13c,v13r,itera=10,efair=0.005)
                oN2 = fair_vectors(oN2,v24l,v24c,v24r,itera=10,efair=0.005) 

            cs13,cs24 = self.mesh.get_isoline_normal_binormal_angles(assign=[v,vN,oN1,oN2])
            return an,oN1,oN2,cs13,cs24
        else:
            num = self.mesh.num_rrv4f4
            arr3 = np.arange(3*num)
            def _get_binormal_cos0(Nps,numpl):
                "if AoTu: oN should be up/down orientation"
                if self.is_pseudogeo_allSameAngle:
                    c_cos, c_sin = Nps-3, Nps-2
                    c_bin = arr3 + Nps-3*num - 3
                else:
                    c_cos = np.arange(numpl) + Nps-numpl*3
                    c_sin = c_cos + numpl
                    c_bin = arr3 + Nps-3*num - numpl*3
                oN = self.X[c_bin].reshape(-1,3,order='F')  
                cos,sin = self.X[c_cos],self.X[c_sin]
                return oN,[cos,sin]
            
            def _get_binormal_cos(Nps,numpl):
                "if AoTu: oN should be up/down orientation"
                if self.is_pseudogeo_allSameAngle:
                    numpl = 1
                    c_cos = Nps-1
                    c_bin = arr3 + Nps-3*num - numpl
                else:
                    c_cos = np.arange(numpl) + Nps-numpl
                    c_bin = arr3 + Nps-3*num - numpl
                oN = self.X[c_bin].reshape(-1,3,order='F')  
                cos = self.X[c_cos]
                return oN,cos
            
            numpl13 = self.mesh.all_rr_polylines_num[0]
            numpl24 = self.mesh.all_rr_polylines_num[1]
            if GO_Ps:
                oN1,cs13 = _get_binormal_cos0(self._Ngops,numpl13)
                oN2,cs24 = _get_binormal_cos0(self._Ngops,numpl24)
                return an,oN1,oN2,cs13,cs24  
            else:
                oN1,cos13 = _get_binormal_cos(self._Nps1,numpl13)
                oN2,cos24 = _get_binormal_cos(self._Nps2,numpl24)
            return an,oN1,oN2,cos13,cos24   
        
    def pseudogeodesic_rectifying_srf(self,width,all_on=None,centerline=False,
                                      is_smooth=False):
        if all_on is not None:
            if centerline:
                "strip's centerline pass through the polyline"
                an = self.mesh.vertices - 0.5*all_on
            else:
                an = self.mesh.vertices
            crvlists1,crvlists2 = self.mesh.continue_family_poly
            ind1 = np.array(list(itertools.chain(*crvlists1)))
            ind2 = np.array(list(itertools.chain(*crvlists2)))
            "need to check crvlist1 or crvlist2"
            arr1,arr2=[],[]
            for ilist in crvlists1:
                arr1.append(len(ilist))
            for ilist in crvlists2:
                arr2.append(len(ilist))   
            arr1,arr2 = np.array(arr1), np.array(arr2)
            sm1 = get_strip_from_rulings(an[ind1],all_on[ind1],arr1,is_smooth)
            sm2 = get_strip_from_rulings(an[ind2],all_on[ind2],arr2,is_smooth)
            return sm1,arr1,sm2,arr2
        else:
            an,on1,on2,_,_ = self.data_pseudogeodesic_binormal
            on1 *= width*2
            on2 *= width*2
            ind1,ind2 = self.mesh.all_rr_polylines_v_vstar_order
            arr1,arr2 = self.mesh.all_rr_polylines_vnum_arr
            if centerline:
                "strip's centerline pass through the polyline"
                an1 = an - 0.5*on1
                an2 = an - 0.5*on2
            else:
                "strip's bottomcrv pass through the polyline"
                an1=an2 = an
    
            sm1 = get_strip_from_rulings(an1[ind1],on1[ind1],arr1,is_smooth)
            sm2 = get_strip_from_rulings(an2[ind2],on2[ind2],arr2,is_smooth)
            return sm1,arr1,sm2,arr2
    
    def pseudogeodesic_rectifying_pq_normal(self,width=1):
        id13l,id13r,id24l,id24r = self.mesh.all_rr_polylines_pq_vstar_order
        #an,on1,on2,_,_ = self.data_pseudogeodesic_binormal#self.pseudogeodesic_binormal()
        an,on1,on2,_,_ = self.pseudogeodesic_binormal()
        def get_planar_quad_normal(idl,idr,on):
            Vl,Vr = an[idl], an[idr]
            oN = on[idl]
            pN = np.cross(Vl-Vr, oN)
            pN = pN / np.linalg.norm(pN,axis=1)[:,None]
            bary = (Vl+Vr+Vl+on[idl]*width+Vr+on[idr]*width)/4
            return bary,pN
        an1,pN1 = get_planar_quad_normal(id13l,id13r,on1)
        an2,pN2 = get_planar_quad_normal(id24l,id24r,on2)
        return [an1,pN1],[an2,pN2]

    def get_pseudogeodesic_constwidth(self,another=False,unique=False):
        "for the initial width in X"
        if another:
            id_ao12tu12 = self.mesh.all_rr_polylines_aotu1234_index[1]
            Nps_width = self._Nps_width2
            num_strip24 = self.mesh.all_rr_aotu_strips_num[1]
            num_width = num_strip24[0]+num_strip24[2]
            arr_AC,arr_DB = self.mesh.all_rr_aotu_strip_width_arr[1]
        else:
            id_ao12tu12 = self.mesh.all_rr_polylines_aotu1234_index[0]
            Nps_width = self._Nps_width1
            num_strip13 = self.mesh.all_rr_aotu_strips_num[0]
            num_width = num_strip13[0]+num_strip13[2]
            arr_AC,arr_DB = self.mesh.all_rr_aotu_strip_width_arr[0]
        "Note: below only for len(A)=len(C)=len(D)=len(B); otherwise need change"
        if self.is_initial: 
            V = self.mesh.vertices
            rrv = self.mesh.ver_rrv4f4
            id1,id2,id3,id4 = id_ao12tu12
            A,C,D,B = V[rrv[id1]],V[rrv[id2]],V[rrv[id3]],V[rrv[id4]]
            "NOTE: HAS PROBLEM FOR different len(A)!=len(C); len(D)!=len(B)"
            ia,ib = min(len(id1),len(id2)), min(len(id3),len(id4))
            widthAC = np.linalg.norm(A[:ia]-C[:ia],axis=1)
            widthDB = np.linalg.norm(D[:ib]-B[:ib],axis=1)
            if unique:
                width = np.mean(np.r_[widthAC,widthDB])
            else:
                "len(arr_AC)+len(arr_DB)=num_width"
                width = np.array([])
                k = 0
                for i in arr_AC:
                    width = np.r_[width,np.mean(widthAC[k:k+i])]
                    k +=i
                k = 0
                for i in arr_DB:
                    width = np.r_[width,np.mean(widthDB[k:k+i])]
                    k +=i
            #print(width,arr_AC,arr_DB,ia,ib)
        else:
            if unique:
                width = self.X[Nps_width-1]
            else:
                c_width = Nps_width - num_width + np.arange(num_width)
                width = self.X[c_width]
        return width
    
    #--------------------------------------------------------------------------
    #                                Errors strings
    #--------------------------------------------------------------------------
    def make_errors(self):
        self.planarity_error()
        self.orthogonal_error()


    def planarity_error(self):
        if self.get_weight('planarity') == 0:
            return None
        P = self.mesh.face_planarity()
        emean = np.mean(P)
        emax = np.max(P)
        self.add_error('planarity', emean, emax, self.get_weight('planarity'))
        print('planarity:[mean,max]=','%.3g'%emean,'%.3g'%emax)

    def orthogonal_error(self):
        if self.get_weight('orthogonal') == 0:
            return None
        _,_,t1,t2,_ = self.mesh.get_quad_midpoint_cross_vectors()
        cos = np.einsum('ij,ij->i',t1,t2)
        cos0 = np.mean(cos)
        err = np.abs(cos-cos0)
        emean = np.mean(err)
        emax = np.max(err)
        self.add_error('orthogonal', emean, emax, self.get_weight('orthogonal'))
        print('orthogonal:[mean,max]=','%.3g'%emean,'%.3g'%emax)

    def anet_error(self):
        if self.get_weight('Anet') == 0 and self.get_weight('Anet_diagnet')==0:
            return None
        if self.get_weight('Anet'):
            name = 'Anet'
            v,v1,v2,v3,v4 = self.mesh.rrv4f4
        elif self.get_weight('Anet_diagnet'):
            name = 'Anet_diagnet'
            v,v1,v2,v3,v4 = self.mesh.rr_star_corner
            
        if self.is_initial:    
            Nv = self.mesh.vertex_normals()[v]
        else:
            num = len(v)
            c_n = self._Nanet-3*num+np.arange(3*num)
            Nv = self.X[c_n].reshape(-1,3,order='F')        
        V = self.mesh.vertices
        err1 = np.abs(np.einsum('ij,ij->i',Nv,V[v1]-V[v]))
        err2 = np.abs(np.einsum('ij,ij->i',Nv,V[v2]-V[v]))
        err3 = np.abs(np.einsum('ij,ij->i',Nv,V[v3]-V[v]))
        err4 = np.abs(np.einsum('ij,ij->i',Nv,V[v4]-V[v]))
        Err = err1+err2+err3+err4
        emean = np.mean(Err)
        emax = np.max(Err)
        self.add_error(name, emean, emax, self.get_weight(name))  
        print('anet:[mean,max]=','%.3g'%emean,'%.3g'%emax)



    def planarity_error_string(self):
        return self.error_string('planarity')

    def orthogonal_error_string(self):
        return self.error_string('orthogonal')
    
    def anet_error_string(self):
        return self.error_string('Anet')