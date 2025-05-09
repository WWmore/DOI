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

from geometrylab.geometry.circle import circle_three_points

from archgeolab.constraints.constraints_basic import con_planarity_constraints

from archgeolab.constraints.constraints_fairness import con_fairness_4th_different_polylines

from archgeolab.constraints.constraints_net import con_unit_edge,con_orient_rr_vn,\
    con_osculating_tangents,con_Gnet,con_CGC,\
    con_orthogonal_midline,con_Anet,con_Snet,con_DOI,con_DOI__freeform,\
    con_Kite,con_Pnet
    
from archgeolab.constraints.constraints_glide import con_glide_in_plane,\
    con_alignment,con_alignments,con_selected_vertices_glide_in_one_plane,\
    con_fix_vertices,con_sharp_corner
                    
from archgeolab.archgeometry.conicSection import interpolate_sphere

from archgeolab.archgeometry.getGeometry import get_strip_from_rulings

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class GP_DOINet(GuidedProjectionBase):
    _mesh = None
    
    _N1 = 0
    
    _N2 = 0

    _N3 = 0

    _N4 = 0
    
    _N5 = 0
    
    _Noscut = 0
    _Norient = 0
    _Ncgc = 0
    _Nanet = 0
    _Nsnet,_Ns_n,_Ns_r = 0,0,0
    _Nsdnet = 0
    
    _Nps1 = 0
    _Nps2 = 0
    _Nps_orient1 = 0
    _Nps_orient2 = 0

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

        'planarity' : 0,

        'orthogonal' :0,
        
        'DOI' :0,
        
        'Kite' :0,
        
        'CGC' :0,
        'Gnet' : 0,  
        'Anet' : 0,
        'Snet' : 0,
        'Snet_orient' : 1,
        'Snet_constR' : 0,

        'Pnet' :0,

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
        
        self.is_GO_or_OG = True
        self.is_diag_or_ctrl = False
        
        self.unit_edge_vec = False
        self.oscu_rrv_tangent = False
        self.orient_rrv_normal = False
        
        self.is_DOI_SIR = False
        self.is_DOI_SIR_diagKite = False
        self.is_Kite_diagGPC = False
        self.is_Kite_diagGPC_SIR = False
        
        ##pseudogeodesic project:
        self.is_pseudogeo_allSameAngle = True ##default is changed to True
        self.is_assigned_angle = False ## if assigned a constant angle
        self.assigned_angle = None

        self.is_pseudogeo_rectify_dvlp = False ## no use
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
                   
                   self.get_weight('orthogonal'),
 
                   self.get_weight('DOI'),
                   
                   self.get_weight('Kite'),
                   
                   self.get_weight('CGC'),
                   self.get_weight('Pnet'),
                   self.get_weight('Anet'),
                   self.get_weight('Snet'),

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
          
    def set_dimensions(self): # Huinote: be used in guidedprojectionbase
        "X:= [Vx,Vy,Vz]"
        V = self.mesh.V
        F = self.mesh.F
        N = 3*V
        N1 = N2 = N3 = N4 = N5 = N
        num_rrstar = self.mesh.num_rrv4f4 ##may have problem for only one strip
        
        Noscut = N
        Norient = N

        Ncgc = N
        Nanet = N
        Nsnet = Ns_n = Ns_r = N
        
        Nps1 = Nps2 = Nps_orient1 = Nps_orient2 = N

        #---------------------------------------------
        if self.get_weight('planarity') != 0:
            "X += [Nx,Ny,Nz]"
            N += 3*F
            N1 = N2 = N3 = N4 = N

        if self.unit_edge_vec: #for Gnet, AGnet; but not for CGC
            "X+=[le1,le2,le3,le4,ue1,ue2,ue3,ue4]"
            "for Anet, AGnet, DGPC"
            N += 16*num_rrstar
            N5 = N
        
        if self.oscu_rrv_tangent: #CGC
            "X +=[ll1,ll2,ll3,ll4,lu1,lu2,u1,u2]"
            N += 12*num_rrstar
            Noscut = N  
        
        if self.orient_rrv_normal: #CGC
            "X+=[vn, a], vN * Nv = a^2>=0; Nv is given orient-vertex-normal"
            N += 4*num_rrstar
            Norient = N

        if self.get_weight('CGC'):
            "X += [Cg1,Cg2, rho_g]"
            N += 6*num_rrstar + 1 #2*num_rrstar
            Ncgc = N

        if self.get_weight('Anet'):
            N += 3*num_rrstar
            Nanet = N

        if self.get_weight('Snet'):
            num_snet = num_rrstar
            N += 11*num_snet  
            Nsnet = N
            if self.get_weight('Snet_orient'):
                N +=4*num_snet  
                Ns_n = N
            if self.get_weight('Snet_constR'):
                N += 1
                Ns_r = N

        if self.get_weight('Pnet'):  
            "based on orient_rr_vn=True"
            if self.is_diag_or_ctrl:
                num13,num24 = self.mesh.all_rr_diag_polylines_num
            else:
                num13,num24 = self.mesh.all_rr_polylines_num

            ###'pseudogeo_1st'
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
            
            # if self.is_pseudogeo_orient:
            N += num_cos + num_rrstar
            Nps_orient1 = N
                
            
            ###'pseudogeo_2nd'
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
            
            #if self.is_pseudogeo_orient:
            N += num_cos + num_rrstar
            Nps_orient2 = N

        #---------------------------------------------
        if N1 != self._N1 or N2 != self._N2:
            self.reinitialize = True
        if N3 != self._N3 or N4 != self._N4:
            self.reinitialize = True
        if self._N2 - self._N1 == 0 and N2 - N1 > 0:
            self.mesh.reinitialize_densities()

        if N5 != self._N5:
            self.reinitialize = True
            
        if Noscut != self._Noscut:
            self.reinitialize = True
        if Norient != self._Norient:
            self.reinitialize = True 
        if Ncgc != self._Ncgc:
            self.reinitialize = True
        if Nanet != self._Nanet:
            self.reinitialize = True
        if Nsnet != self._Nsnet:
            self.reinitialize = True
        if Ns_n != self._Ns_n:
            self.reinitialize = True
        if Ns_r != self._Ns_r:
            self.reinitialize = True
           
        if Nps1 != self._Nps1:
            self.reinitialize = True  
        if Nps2 != self._Nps2:
            self.reinitialize = True 
        if Nps_orient1 != self._Nps_orient1:
            self.reinitialize = True  
        if Nps_orient2 != self._Nps_orient2:
            self.reinitialize = True  

        #----------------------------------------------
        self._N = N
        self._N1 = N1
        self._N2 = N2
        self._N3 = N3
        self._N4 = N4
        self._N5 = N5
        self._Noscut = Noscut        
        self._Norient = Norient
        self._Ncgc = Ncgc
        self._Nanet = Nanet
        self._Nsnet,self._Ns_n,self._Ns_r = Nsnet,Ns_n,Ns_r
        
        self._Nps1 = Nps1
        self._Nps2 = Nps2
        self._Nps_orient1 = Nps_orient1
        self._Nps_orient2 = Nps_orient2
        
        self.build_added_weight() # Hui add
        
        
    def initialize_unknowns_vector(self):
        "X:= [Vx,Vy,Vz]"
        X = self.mesh.vertices.flatten('F')
        if self.get_weight('planarity') != 0:
            "X += [Nx,Ny,Nz]; len=3F"
            normals = self.mesh.face_normals()
            X = np.hstack((X, normals.flatten('F')))
            
        if self.unit_edge_vec:
            _,l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_unit_edge(self.is_diag_or_ctrl)
            X = np.r_[X,l1,l2,l3,l4]
            X = np.r_[X,E1.flatten('F'),E2.flatten('F'),E3.flatten('F'),E4.flatten('F')]
        
        ### CGC: 
        if self.oscu_rrv_tangent:
            "X +=[ll1,ll2,ll3,ll4,lu1,lu2,u1,u2]"
            l,t,lt1,lt2 = self.mesh.get_net_osculating_tangents(self.is_diag_or_ctrl)
            [ll1,ll2,ll3,ll4],[lt1,t1],[lt2,t2] = l,lt1,lt2
            X = np.r_[X,ll1,ll2,ll3,ll4]
            X = np.r_[X,lt1,lt2,t1.flatten('F'),t2.flatten('F')] 

        if self.orient_rrv_normal:
            _,vN,a = self.mesh.get_v4_orient_unit_normal(self.is_diag_or_ctrl)
            X = np.r_[X,vN.flatten('F'),a]     
            
        if self.get_weight('CGC'):
            "X += [Cg1,Cg2, rho_g], Cg1 geodesic circle centers from isoline1, rho_g is 1/kappa_g"
            Cg1, Cg2, rho = self.get_geodesic_curvature(self.is_diag_or_ctrl)
            X = np.r_[X,Cg1.flatten('F'),Cg2.flatten('F'), rho]  
            
        ### CNC:
        if self.get_weight('Anet'):
            if self.get_weight('Anet'):
                if True:
                    "only r-regular vertex"
                    v = self.mesh.ver_rrv4f4
                else:
                    v = self.mesh.ver_regular
            V4N = self.mesh.vertex_normals()[v]
            X = np.r_[X,V4N.flatten('F')]
        
        if self.get_weight('Snet'):
            r = self.get_weight('Snet_constR')
            x_snet,_ = self.get_snet(r,self.is_diag_or_ctrl)
            X = np.r_[X,x_snet]

        ### Pnet: ##merge pseudogeo_1st and pseudogeo_2nd to Pnet
        if self.get_weight('Pnet'):
            """ based on orient_rr_vs: X +=[vN,a] (vN*Nv=a^2)
            X=+[oN1; cos1; a,b; width; pn; xy]
            if orientT(default): oN1,oN2 are oriented with t1,t2 = e1-e3, e2-e4
            if orientN: oN is oriented with the oriented-vertex-normal
            """
            if self.data_pseudogeodesic_binormal is None:
                "Pl. Click &  Check binormals first: "
                _,oN1,oN2,cs13,cs24 = self.pseudogeodesic_binormal(self.is_diag_or_ctrl)
            else:
                _,oN1,oN2,cs13,cs24 = self.data_pseudogeodesic_binormal
            cos1,cos2 = np.abs(cs13[0]), np.abs(cs24[0])
            #_,_,_,_,_,e1,e2,e3,e4 = self.mesh.get_v4_unit_edge(True)
            _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents(self.is_diag_or_ctrl)
            vN = self.mesh.get_v4_unit_normal(self.is_diag_or_ctrl)[2] ##vN:=ut1xut2/||
            
            ###if self.get_weight('pseudogeo_1st'):
            if self.is_pseudogeo_allSameAngle:
                cos1 = np.mean(cos1)
            X = np.r_[X,oN1.flatten('F'), cos1]
            
            #if self.is_pseudogeo_orient:
            "vn*on1=cos1=a1^2;on1*(vNxut1)=sin1=b1^2; a1^4+b1^4==1"
            a = np.sqrt(np.abs(cos1))
            U = np.cross(vN,ut1)/np.linalg.norm(np.cross(vN,ut1),axis=1)[:,None]
            b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN1,U)))
            #"vn*on1=cos1=a1^2;on1*(e2-e4)=b1^2"
            #b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN1,e2-e4)))
            X = np.r_[X,a,b]                


            ###if self.get_weight('pseudogeo_2nd'):
            if self.is_pseudogeo_allSameAngle:
                cos2 = np.mean(cos2)
            X = np.r_[X,oN2.flatten('F'), cos2]
            
            #if self.is_pseudogeo_orient:
            "either or both family(ies) have oriented binormals between <vn,t2(t1)>"
            "vn*on1=cos1=a1^2;on1*(vNxut1)=sin1=b1^2; a1^4+b1^4==1"
            a = np.sqrt(np.abs(cos2))
            U = np.cross(vN,ut2)/np.linalg.norm(np.cross(vN,ut2),axis=1)[:,None]
            b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN2,U)))
            #"vn*on2=cos2=a2^2;on2*(e1-e3)=b2^2"
            #b = np.sqrt(np.abs(np.einsum('ij,ij->i',oN2,e1-e3)))
            X = np.r_[X,a,b]
                     
        #-----------------------
        
        self._X = X
        self._X0 = np.copy(X)
            
        self.build_added_weight() # Hui add


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
            H,r = con_fairness_4th_different_polylines(pl1,is_diagnet=True,**self.weights)
            self.add_iterative_constraint(H, r, 'fairness_diag_4diff')     
        
        if self.get_weight('planarity'):
            #"use Hui's way not Davide's"
            H,r = con_planarity_constraints(**self.weights)  
            self.add_iterative_constraint(H, r, 'planarity')

            
        ###-------partially shared-used codes:---------------------------------
        if self.get_weight('boundary_z0') !=0:
            z = 0
            v = np.array([816,792],dtype=int)
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

        if self.unit_edge_vec: 
            H,r = con_unit_edge(self.is_diag_or_ctrl,**self.weights)
            self.add_iterative_constraint(H, r, 'unit_edge')

        if self.get_weight('Gnet'):
            H,r = con_Gnet(**self.weights)
            self.add_iterative_constraint(H, r, 'Gnet')   
            
        if self.oscu_rrv_tangent:
            H,r = con_osculating_tangents(self.is_diag_or_ctrl,**self.weights)
            self.add_iterative_constraint(H, r, 'oscu_rrv_tangent')
        
        if self.orient_rrv_normal:
            "rr_vn orients samely as Nv"
            is_oscut = True if self.oscu_rrv_tangent else False
            H,r = con_orient_rr_vn(is_oscut, **self.weights)
            self.add_iterative_constraint(H, r, 'orient_vn')
            
        if self.get_weight('orthogonal'):
            H,r = con_orthogonal_midline(**self.weights)
            self.add_iterative_constraint(H, r, 'orthogonal')
        
        if self.get_weight('DOI'):
            yes1, yes2 = self.is_DOI_SIR, self.is_DOI_SIR_diagKite
            if True:
                H,r = con_DOI__freeform(self.is_GO_or_OG,yes1,yes2,**self.weights)
            else:
                "works well, but only for patch or rotational mesh"
                H,r = con_DOI(self.is_GO_or_OG,yes1,**self.weights)
            self.add_iterative_constraint(H, r, 'DOI')
            
        if self.get_weight('Kite'):
            yes1,yes2 = self.is_Kite_diagGPC, self.is_Kite_diagGPC_SIR
            H,r = con_Kite(self.is_GO_or_OG,yes1,yes2,**self.weights)
            self.add_iterative_constraint(H, r, 'Kite')
        #elif self.get_weight('Kite_diagnet'): #Hui: works but not use in alignnet on SIR-net
            #H,r = con_kite_diagnet(self.is_GO_or_OG,**self.weights)
            #self.add_iterative_constraint(H, r, 'Kite_diagnet')
            
        if self.get_weight('CGC'):
            H,r = con_CGC(self.is_diag_or_ctrl,**self.weights)
            self.add_iterative_constraint(H, r, 'CGC')
            
        if self.get_weight('Anet'):
            H,r = con_Anet(is_diagnet=self.is_diag_or_ctrl,**self.weights)
            self.add_iterative_constraint(H, r, 'Anet')

        if self.get_weight('Snet'):
            orientrn = self.mesh.new_vertex_normals()
            H,r = con_Snet(orientrn,
                           is_diagnet=self.is_diag_or_ctrl,
                           is_uniqR=self.if_uniqradius,
                           assigned_r=self.assigned_snet_radius,
                           **self.weights)
            self.add_iterative_constraint(H, r, 'Snet') 

        ## from initial net-pattern to get pseudogeodesic-pattern:
        if self.get_weight('Pnet'):
            "based on self.orient_rr_vn = True"
            is_diagnet = self.is_diag_or_ctrl
            is_dev = self.is_pseudogeo_rectify_dvlp
            is_unique_angle, coss = False, None

            "along each curve, angle is const. but different from each other"
            if self.is_assigned_angle:
                "there is uniqe const. angle"
                is_unique_angle=True
                coss = np.cos(self.assigned_angle/180.0*np.pi)

            H,r = con_Pnet(is_diagnet,is_dev,is_unique_angle,coss,**self.weights)     
            self.add_iterative_constraint(H, r, 'Pnet')
            
            
        # if self.get_weight('pseudogeo_1st') or self.get_weight('pseudogeo_2nd'):
        #     "based on self.orient_rr_vn = True"
        #     vN = self.mesh.get_v4_orient_unit_normal()[1] 
        #     is_unique_angle, coss = False, None
        #     "along each curve, angle is const. but different from each other"
            
        #     if self.is_pseudogeo_allSameAngle:
        #         "there is uniqe const. angle"
        #         is_unique_angle=True
        #         coss = np.cos(self.pseudogeo_1st_constangle/180.0*np.pi)
                
        #     if self.is_diag_or_ctrl:
        #         T1 = self.mesh.get_v4_diag_unit_tangents()[2]
        #         T2 = self.mesh.get_v4_diag_unit_tangents()[3]
        #     else:
        #         T1 = self.mesh.get_v4_unit_tangents()[2]
        #         T2 = self.mesh.get_v4_unit_tangents()[3]

        #     U1 = np.cross(vN, T1) / np.linalg.norm(np.cross(vN, T1),axis=1)[:,None]
        #     U2 = np.cross(vN, T2) / np.linalg.norm(np.cross(vN, T2),axis=1)[:,None]
        #     U1xyz, U2xyz = U1.flatten('F'), U2.flatten('F')
            
        #     H,r = con_Pnet(U1xyz,U2xyz,self.is_diag_or_ctrl,
        #                    is_unique_angle,coss,**self.weights)
        #     self.add_iterative_constraint(H, r, 'Pnet') 
     
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
        
        self.add_weight('Noscut', self._Noscut)
        self.add_weight('Norient', self._Norient)
        self.add_weight('Ncgc', self._Ncgc)
        self.add_weight('Nanet', self._Nanet)
        self.add_weight('Nsnet', self._Nsnet)
        self.add_weight('Ns_n', self._Ns_n)
        self.add_weight('Ns_r', self._Ns_r)
        
        self.add_weight('Nps1', self._Nps1)
        self.add_weight('Nps2', self._Nps2)
        self.add_weight('Nps_orient1', self._Nps_orient1)
        self.add_weight('Nps_orient2', self._Nps_orient2)


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
    #                       Getting (initilization + Plotting):
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

    def get_osculating_tangents(self):
        if self.is_initial:
            X = self._X0
        else:
            X = self.X
        v = self.mesh.ver_rrv4f4
        an = self.mesh.vertices[v]
        num = self.mesh.num_rrv4f4
        arr3 = np.arange(3*num)
        s = self._Noscut - 12*num
        c_t1 = s+6*num+arr3
        c_t2 = s+9*num+arr3        
        t1 = X[c_t1].reshape(-1,3,order='F')
        #ut1 = t1 / np.linalg.norm(t1,axis=1)[:,None]
        t2 = X[c_t2].reshape(-1,3,order='F')
        #ut2 = t2 / np.linalg.norm(t2,axis=1)[:,None]
        return an,t1,t2  
    
    def get_orient_rr_normal(self,is_diagnet=False,initialized=True):
        if initialized or self.is_initial or not self.orient_rrv_normal:
            return self.mesh.get_v4_orient_unit_normal(is_diagnet) ##==[an,vN,a]
        elif self.orient_rrv_normal:
            v = self.mesh.ver_rrv4f4
            an = self.mesh.vertices[v]
            num = self.mesh.num_rrv4f4
            c_n = np.arange(3*num) + self._Norient-4*num 
            vN = self.X[c_n].reshape(-1,3,order='F')
            return [an,vN,0]
        
    def get_geodesic_curvature(self,is_diagnet=False):
        """ curvature \kappa that projects on tangent plane and surface normal 
        leads to geodesic curvature \kappa_g and normal curvature \kappa_n,
        whose inversions are radii of corresponding circles: 
            rho_g = 1/\kappa_g; rho_n = 1/\kappa_n
        where the geodesic circle center Cg is on tangent plane.
        
        For v1-v-v3,there exists Cg1; for v2-v-v4, there exists Cg2;
        CGC-net is defined by two families of isolines with constant |\kappa_g1=|\kappa_g2|,
        that is unique rho_g = 1/\kappa_g for both isolines
        
        X +=[Cg1, Cg2, rho_g], len(Cg1)=len(Cg2)=3*num_rrv4f4, len(rho_g)=1
        for isoline1 with v1-v-v3:
            orientVN * (Cg1-V) = 0, (rho_g)^2=(Cg1-V)^2=(Cg1-V1)^2=(Cg1-V3)^2
        for isoline2 with v2-v-v4:
            orientVN * (Cg2-V) = 0, (rho_g)^2=(Cg2-V)^2=(Cg2-V2)^2=(Cg2-V4)^2
        """
        V = self.mesh.vertices
        if is_diagnet:
            v0,v1,v2,v3,v4 = self.mesh.rr_star_corner
        else:
            v0,v1,v2,v3,v4 = self.mesh.rrv4f4
        
        _,vN,_ = self.mesh.get_v4_orient_unit_normal(is_diagnet)
        
        eps = np.finfo(float).eps
        
        def _get_center(v0,v1,v3):
            _,O1,r1 = circle_three_points(V[v1],V[v0],V[v3], center=True)

            kd1 = (O1-V[v0]) / (r1[:,None]+eps) ##unit curvature vector
            T1 = np.cross(vN,np.cross(kd1,vN+eps)) ##note: if geodesic, then O1-V[v0] // vN
            T1 = T1 / (np.linalg.norm(T1,axis=1)[:,None]+eps)

            cos1 = np.einsum('ij,ij->i', kd1, T1)
            rho1 = r1 / (cos1+eps) 
            Cg1 = V[v0] + T1 * rho1[:,None]
            return Cg1, rho1
            
        Cg1,rho1 = _get_center(v0, v1, v3)
        Cg2,rho2 = _get_center(v0, v2, v4)
        
        print('rho1[min,max]=',np.min(rho1), np.max(rho1))
        print('rho2[min,max]=',np.min(rho2), np.max(rho2))
        
        rho = np.mean(np.r_[rho1,rho2])
        
        return Cg1, Cg2, rho #rho1,rho2 ##used for checking


    def get_CGC_circular_strip(self,width,is_diagnet=False,
                               is_centerline=False,is_smooth=False):
        v = self.mesh.ver_rrv4f4
        an = self.mesh.vertices[v]
        if self.is_initial: 
            Cg1, Cg2, rho = self.get_geodesic_curvature(is_diagnet)
        else:
            num = len(v)
            c_cg1 = self._Ncgc - 6*num - 1 + np.arange(3*num)
            c_cg2 = c_cg1 + 3*num
            Cg1 = self.X[c_cg1].reshape(-1,3,order='F')
            Cg2 = self.X[c_cg2].reshape(-1,3,order='F') 

        T1, T2 = Cg1 - an, Cg2 - an
        eps = np.finfo(float).eps 
        unitT1 = T1 / (np.linalg.norm(T1,axis=1)[:,None]+eps)
        unitT2 = T2 / (np.linalg.norm(T2,axis=1)[:,None]+eps)
        T1, T2 = unitT1 * width, unitT2 * width

        if is_centerline:
            "strip's centerline pass through the polyline"
            an1 = an - 0.5*T1
            an2 = an - 0.5*T2
        else:
            "strip's bottomcrv pass through the polyline"
            an1=an2 = an

        if is_diagnet:
            ind1,ind2 = self.mesh.all_rr_diag_polylines_v_vstar_order
            arr1,arr2 = self.mesh.all_rr_diag_polylines_vnum_arr
        else:
            ind1,ind2 = self.mesh.all_rr_polylines_v_vstar_order
            arr1,arr2 = self.mesh.all_rr_polylines_vnum_arr
    
        sm1 = get_strip_from_rulings(an1[ind1],T1[ind1],arr1,is_smooth)
        sm2 = get_strip_from_rulings(an2[ind2],T2[ind2],arr2,is_smooth)
        return sm1,arr1,sm2,arr2

    def get_CNC_circular_strip(self,width,is_diagnet=False,
                               is_centerline=False,is_smooth=False):
        v = self.mesh.ver_rrv4f4
        an = self.mesh.vertices[v]     
        if self.is_initial: 
            centers,_ = self.get_snet_data(is_diagnet,center=True)
        else:
            num = len(v)
            _n1 = self._Nsnet-11*num
            arr1 = np.arange(num)
            c_a = _n1+5*num+arr1
            c_b,c_c,c_d = c_a+num,c_a+2*num,c_a+3*num
            "sphere center C:= (m1,m2,m3) = -(b, c, d) /a/2"
            a,b,c,d = self.X[c_a],self.X[c_b],self.X[c_c],self.X[c_d]
            centers = -0.5*np.c_[b/a,c/a,d/a]
        
        ## below is same as get_CGC_circular_strip
        eps = np.finfo(float).eps 
        unitN = (an-centers) / (np.linalg.norm(an-centers,axis=1)[:,None]+eps)
        N = unitN * width
        
        if is_centerline:
            "strip's centerline pass through the polyline"
            an1 = an - 0.5*N
            an2 = an - 0.5*N
        else:
            "strip's bottomcrv pass through the polyline"
            an1=an2 = an

        if is_diagnet:
            ind1,ind2 = self.mesh.all_rr_diag_polylines_v_vstar_order
            arr1,arr2 = self.mesh.all_rr_diag_polylines_vnum_arr
        else:
            ind1,ind2 = self.mesh.all_rr_polylines_v_vstar_order
            arr1,arr2 = self.mesh.all_rr_polylines_vnum_arr
            
        sm1 = get_strip_from_rulings(an1[ind1],N[ind1],arr1,is_smooth)
        sm2 = get_strip_from_rulings(an2[ind2],N[ind2],arr2,is_smooth)
        return sm1,arr1,sm2,arr2        

    def get_snet(self,is_r,is_diagnet=False,is_orient=True):
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
        if is_diagnet:
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

    def get_snet_data(self,is_diagnet=False,
                      center=False,normal=False,tangent=False,ss=False,
                      is_diag_binormal=False):
        "at star = self.rr_star"
        V = self.mesh.vertices
        if is_diagnet:
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
            if is_diagnet:
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
  
    #------------------------------------------------------------------------

    def pseudogeodesic_binormal(self, is_initial=False, is_diagnet=False, ##is_diagnet is newly added
                                is_orientT=False,is_orientN=False,
                                is_remedy=False, is_smooth=False):
        "save as self.data_pseudogeodesic_binormal"
        V = self.mesh.vertices
        v = self.mesh.ver_rrv4f4   
        an, num = V[v], len(v)
        if self.is_initial or is_initial: 
            
            if is_diagnet:
                v,v1,v2,v3,v4 = self.mesh.rr_star_corner
            else:
                v,v1,v2,v3,v4 = self.mesh.rrv4f4

            vN = self.mesh.get_v4_orient_unit_normal(is_diagnet)[1]
            _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents(is_diagnet)
            
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

            cs13,cs24 = self.mesh.get_isoline_normal_binormal_angles(is_diagnet, assign=[v,vN,oN1,oN2])
            return an,oN1,oN2,cs13,cs24
        else:
            "computed from optimization variables self.X: "
            arr3 = np.arange(3*num)

            def _get_binormal_cos(Nps,numpl):
                "if AoTu: oN should be up/down orientation"
                if self.is_pseudogeo_allSameAngle:##True in default
                    numpl = 1
                    c_cos = Nps-1
                    c_bin = arr3 + Nps-3*num - numpl
                else:
                    c_cos = np.arange(numpl) + Nps-numpl
                    c_bin = arr3 + Nps-3*num - numpl
                oN = self.X[c_bin].reshape(-1,3,order='F')  
                cos = self.X[c_cos]
                return oN,cos
            
            if is_diagnet: ##newly added
                numpl13,numpl24 = self.mesh.all_rr_diag_polylines_num
            else:
                numpl13,numpl24 = self.mesh.all_rr_polylines_num

            oN1,cos13 = _get_binormal_cos(self._Nps1,numpl13)
            oN2,cos24 = _get_binormal_cos(self._Nps2,numpl24)
            return an,oN1,oN2,cos13,cos24   
        
    def pseudogeodesic_rectifying_srf(self,width,all_on=None,is_diagnet=False,
                                      is_centerline=False,
                                      is_smooth=False):
        if all_on is not None:
            if is_centerline:
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
            ## below is same as get_CGC_circular_strip,get_CNC_circular_strip
            an,on1,on2,_,_ = self.data_pseudogeodesic_binormal
            n1 = on1 * width
            n2 = on2 * width
            if is_centerline:
                "strip's centerline pass through the polyline"
                an1 = an - 0.5*n1
                an2 = an - 0.5*n2
            else:
                "strip's bottomcrv pass through the polyline"
                an1=an2 = an
                
            if is_diagnet:
                ind1,ind2 = self.mesh.all_rr_diag_polylines_v_vstar_order
                arr1,arr2 = self.mesh.all_rr_diag_polylines_vnum_arr
            else:
                ind1,ind2 = self.mesh.all_rr_polylines_v_vstar_order
                arr1,arr2 = self.mesh.all_rr_polylines_vnum_arr
    
            sm1 = get_strip_from_rulings(an1[ind1],n1[ind1],arr1,is_smooth)
            sm2 = get_strip_from_rulings(an2[ind2],n2[ind2],arr2,is_smooth)
            return sm1,arr1,sm2,arr2

    
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
        if self.get_weight('Anet') == 0:
            return None
        if self.get_weight('Anet'):
            
            if self.is_diag_or_ctrl:
                name = 'Anet_diagnet'
                v,v1,v2,v3,v4 = self.mesh.rr_star_corner
            else:
                name = 'Anet'
                v,v1,v2,v3,v4 = self.mesh.rrv4f4

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
        print(name+':[mean,max]=','%.3g'%emean,'%.3g'%emax)



    def planarity_error_string(self):
        return self.error_string('planarity')

    def orthogonal_error_string(self):
        return self.error_string('orthogonal')
    
    def anet_error_string(self):
        return self.error_string('Anet')