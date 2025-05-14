"""
Created on Thu Apr 24 11:13:42 2025

@author: wanghui
"""

#------------------------------------------------------------------------------
import os

import sys

#path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path)
#print(path, os.path)

from traits.api import Button,String,on_trait_change, Float, Bool, Range,Int

from traitsui.api import View, Item, HGroup, Group, VGroup

import numpy as np
#------------------------------------------------------------------------------

from geometrylab.gui.geolabcomponent import GeolabComponent
from geometrylab.vtkplot.edgesource import Edges
from geometrylab.vtkplot.facesource import Faces
from geometrylab.geometry import Polyline

from doi__3_opt import GP_DOINet
from archgeolab.archgeometry.conicSection import get_sphere_packing,\
    get_vs_interpolated_sphere

#------------------------------------------------------------------------------

''' build:  
    show:   
'''

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        InteractiveGuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class DOINet(GeolabComponent):

    name = String('DOI-Net')
    
    itera_run = Int(5)

    epsilon = Float(0.001, label='dumping')

    step = Float(1)

    fairness_reduction = Range(low=0,high=5,value=0,label='F-reduce')

    mesh_fairness = Float(0.0000,label='meshF')

    tangential_fairness = Float(0.0000,label='tangF')

    boundary_fairness = Float(0.0000,label='bondF')

    spring_fairness = Float(0.0000,label='springF')
    
    corner_fairness = Float(0,label='cornerF')
    
    fairness_diagmesh = Float(0,label='diagF')

    reference_closeness = Float(0,label='refC')
    
    fairness_4diff = Float(0,label='Fair4diff')
    fairness_diag_4diff = Float(0,label='FairDiag4diff')

    boundary_glide = Float(0,label='bdryGlide') ##weight for all
    i_boundary_glide = Float(0,label='iBdryGlide') ##weight for i-th
    glide_1st_bdry = Bool(label='1st')
    glide_2nd_bdry = Bool(label='2nd')
    glide_3rd_bdry = Bool(label='3rd')
    glide_4th_bdry = Bool(label='4th')
    glide_5th_bdry = Bool(label='5th')
    glide_6th_bdry = Bool(label='6th')
    glide_7th_bdry = Bool(label='7th')
    glide_8th_bdry = Bool(label='8th')

    show_corner = Bool(label='ShowConer')
    sharp_corner = Bool(label='SharpCor')
    
    self_closeness = Float(0,label='selfC')
    
    set_refer_mesh = Bool(label='SetRefer')
    show_refer_mesh = Bool(label='ShowRefer')
    show_ref_mesh_boundary = Bool(label='ShowReferBdry')
    
    fair0 = Button(label='0')
    fair1 = Button(label='0.1')
    fair01 = Button(label='0.01')
    fair001 = Button(label='0.001')
    fair0001 = Button(label='0.0001')

    close0 = Button(label='0')
    close005 = Button(label='0.005')
    close01 = Button(label='0.01')
    close05 = Button(label='0.05')
    close1 = Button(label='0.1')
    close5 = Button(label='0.5')

    weight_fix = Float(10)
    fix_all = Bool(label='Fix')
    fix_boundary = Bool(label='FixB')
    fix_boundary_i = Bool(label='FixBi')
    fix_corner = Bool(label='FixC')
    fix_p_weight = Float(0,label='Fix_p')
    fix_button = Button(label='Fix')
    unfix_button = Button(label='Unfix')
    clearfix_button = Button(label='Clear')

    boundary_z0 = Bool(label='BZ0')
    selected_z0 = Bool(label='S_Z0')
    selected_y0 = Bool(label='S_Y0')
    z0 = Float(0)#Bool(label='Z=0')
    
    reinitialize = Button(label='ini')
    optimize = Button(label='Opt')
    interactive = Bool(False, label='InteractiveOpt')
    hide_face = Bool(label='HideF')
    hide_edge = Bool(label='HideE')    
    ####----------------------------------------------------------------------- 
    #--------------Optimization: -----------------------------
    button_clear_constraint = Button(label='Clear')
    
    orthogonal = Bool(label='Orthogonal')
    
    GPC_net = Bool(label='GPC')
    
    orient_rrv_normal = Bool(False,label='orientN')
    
    switch_GO_or_OG = Bool(True,label='_GO|OG_')  ## need to choose when window opens; choose geodesic-isoline direction
    switch_diag_or_ctrl = Bool(False,label='_Diag|Ctrl_') ## need to choose when window opens; choose on diagonal net or control net
    switch_kite_1_or_2 = Bool(label='_Kite1|2_')## switch between "|va-v|=|vd-v|.." or "|va-v|=|vc-v|.."
    
    DOI_net = Bool(label='DOI')
    is_DOI_SIR = Bool(label='is_DOI-SIR')
    is_DOI_SIR_diagKite = Bool(label='is_DOI-SIR-diagKite')
    
    Kite_diagnet = Bool(label='Kite_diag')
    Kite_net = Bool(label='Kite')
    is_Kite_diagGPC = Bool(label='is_Kite-GPC')
    is_Kite_diagGPC_SIR = Bool(label='is_Kite-GPC-SIR')
    

    button_align_Kite = Button(label='A-Kite')
    button_align_CGC = Button(label='A-CGC')
    button_align_CNC = Button(label='A-CNC')
    button_align_Pnet = Button(label='A-Pnet')
    
    CGC_net = Bool(label='CGC')
    Gnet = Bool(label='Gnet') 
      
    CNC_net = Bool(label='CNC') ##Snet with const.r
    Snet = Bool(label='Snet')
    Snet_orient = Bool(True,label='Orient')
    Snet_constR = Bool(False,label='constR')
    if_uniqR = Bool(False) 
    Snet_constR_assigned = Float(label='const.R')
    button_CMC_mesh = Button(label='CMC')
    
    Anet = Bool(label='Anet')  
    button_minimal_mesh = Button(label='Minimal')
    
    oscu_rrv_tangent = Bool(False,label='oscuT')
    
    switch_1st_or_2nd = Bool(False,label='_1st|2nd_')
    is_both = Bool(label='is_both')#TODO    

    ##Pseudogeodesic-net:
    Pseudogeodesic_net = Bool(label='Pnet')
    #pseudogeo_rectify_dvlp = Bool(label='Develop')
    #pseudogeo_allSameAngle = Bool(label='uniqAngle')##default is True
    is_assigned_angle = Bool(label='Assigned')
    assigned_angle = Float(60)## if is_assigned_angle=True, const.angle of <normal,tangentplane>
    

    #--------------Plotting: -----------------------------
    show_oscu_tangent = Bool(label='oscuT')
    
    show_cgc_centers = Bool(label='cgcCenters')
    
    show_midpoint_edge1 = Bool(label='E1')
    show_midpoint_edge2 = Bool(label='E2')
    show_midpoint_polyline1 = Bool(label='Ply1')
    show_midpoint_polyline2 = Bool(label='Ply2')
    show_midline_mesh = Bool(label='ReMesh')
    
    show_diagonal_red_mesh = Bool(False,label='DiagM1')
    show_diagonal_blue_mesh = Bool(False,label='DiagM2')
    show_diagonal_mesh = Bool(False,label='DiagM')
    
    show_revolution = Bool(False,label='Rot')

    show_vs_sphere = Bool(label='VS-Sphere')
    show_snet_center = Bool(label='Snet-C')
    show_snet_normal = Bool(label='Snet-N')
    show_snet_tangent = Bool(label='Snet-T')
    
    show_orient_vn = Bool(label='~vN')
    
    show_isoline = Bool(label='Crv')
    
    show_CGC_strip = Bool(label='CGCstrip')
    show_CNC_strip = Bool(label='CNCstrip')
    
    show_pseudogeo_binormal = Bool(label='Pnet-BiN')
    show_Pnet_rectifystrip = Bool(label='Pstrip')
    show_isolinestrip_unroll = Bool(label='Unroll')
    
    strip_width = Float(0.5,label='Width')
    is_central_strip = Bool(True,label='_Central_')
    is_orient_tangent = Bool(True,label='_OrientT_')
    is_orient_normal = Bool(True,label='_OrientN_')
    is_remedied_BiN = Bool(True,label='_remedyBiN_')
    is_smoothed_BiN = Bool(False,label='_SmoothBiN_')
 
    dist_inverval = Float(1.3,label='Dist')
    set_unroll_strip_fairness = Float(0.005,label='Fair')
    is_unroll_midaxis = Bool(True,label='_Mid_')
    
    #--------------Save: --------------------
    save_button = Button(label='Save')
    label = String('obj')
    save_new_mesh = None
    
    #--------------Print: -----------------------------
    print_check = Button(label='Check')
    print_computation = Button(label='Computation')
    print_error = Button(label='Error')
    #--------------------------------------------------------------------------
    view = View(VGroup(Group(
    #---------------------------------------------------------
    Group(## 1st-panel
        VGroup(
              HGroup('switch_GO_or_OG', 'switch_diag_or_ctrl','switch_kite_1_or_2',
                     'oscu_rrv_tangent',
                     'orient_rrv_normal',
                     ),    
              #HGroup('orthogonal','GPC_net'),
              
              VGroup(
                      HGroup('DOI_net','is_DOI_SIR','is_DOI_SIR_diagKite'),
                      HGroup('Kite_net','is_Kite_diagGPC','is_Kite_diagGPC_SIR'),
              show_border=True),
              
              VGroup(
                      HGroup('CGC_net','Gnet',
                             'CNC_net','Anet',),
                      # HGroup('Snet',
                      #        #'Snet_orient',
                      #        'Snet_constR',
                      #        Item('if_uniqR',show_label=False),
                      #        'Snet_constR_assigned'),
                      HGroup('Pseudogeodesic_net',
                             'is_assigned_angle',
                             Item('assigned_angle',show_label=False)),
              show_border=True),
                     
              HGroup(Item('button_align_Kite',show_label=False),
                     Item('button_align_CGC',show_label=False),
                     Item('button_align_CNC',show_label=False),
                     Item('button_align_Pnet',show_label=False),
                     Item('button_clear_constraint',show_label=False)
                     ),
              #HGroup(#Item('button_CMC_mesh',show_label=False),
                     #Item('button_minimal_mesh',show_label=False),
                     #),

        label='Opt.Net',show_border=True),
        #------------------------------------------------  
        Group(
            VGroup(HGroup(#'show_midpoint_edge1',
                          #'show_midpoint_edge2',
                          'show_midpoint_polyline1',
                          'show_midpoint_polyline2',
                          #'show_midline_mesh',
                          'show_diagonal_red_mesh',
                         'show_diagonal_blue_mesh',
                         'show_diagonal_mesh',
                         'show_revolution',
                         ),
                  ##CGC
                  HGroup('show_oscu_tangent',
                         'show_orient_vn',
                         'show_cgc_centers',
                         ),
                  
                  ##CNC
                  HGroup('show_vs_sphere',
                         'show_snet_center',
                         'show_snet_tangent',
                         'show_snet_normal',),
              label='Point / Poly / Mesh',show_border=False),
            
              ###-------------------------------------
              ##Pnet
              VGroup(
                  HGroup('switch_1st_or_2nd','is_both',
                         'show_isoline',),
                  HGroup('is_orient_tangent',
                         'is_orient_normal',
                         'is_remedied_BiN',
                         'is_smoothed_BiN'),
                  HGroup('show_orient_vn','show_pseudogeo_binormal',
                         'show_Pnet_rectifystrip',
                         'show_CGC_strip',
                         'show_CNC_strip',),
                    ##unrollment
                    HGroup('show_isolinestrip_unroll',
                           'strip_width','is_central_strip'),
                    HGroup('dist_inverval',
                           'set_unroll_strip_fairness',
                           'is_unroll_midaxis'),
                label='Strip',show_border=False),
        #------------------------------------------------  
        label='Plotting',show_border=True,layout='tabbed'),
        #------------------------------------------------  
        label='Opt',show_border=False),
    #---------------------------------------------------------
    #---------------------------------------------------------
    Group(## 2nd-panel
        HGroup('itera_run',#'epsilon','step'
               ),
        VGroup(HGroup(Item('fair1',show_label=False),
                      Item('fair01',show_label=False),
                      Item('fair001',show_label=False),
                      Item('fair0001',show_label=False),
                      Item('fair0',show_label=False)),
               HGroup('mesh_fairness',
                      'boundary_fairness',
                      'corner_fairness'),
               HGroup('tangential_fairness',
                      'spring_fairness'),
               HGroup('fairness_4diff',
                      'fairness_diag_4diff'),
                      'fairness_reduction',
                      'fairness_diagmesh',
               HGroup('show_corner','sharp_corner'),
               show_border=True,label='Fairness'),
        ###-------------------------------------
        VGroup(HGroup(Item('close5',show_label=False),
                      Item('close1',show_label=False),
                      Item('close05',show_label=False),
                      Item('close01',show_label=False),
                      Item('close0',show_label=False)),
               HGroup('self_closeness',
                      'reference_closeness'),
                      'boundary_glide',
               HGroup('set_refer_mesh',
                      'show_refer_mesh',
                      'show_ref_mesh_boundary',),
               HGroup('i_boundary_glide',
                      'glide_1st_bdry',
                      'glide_2nd_bdry',
                      'glide_3rd_bdry',
                      'glide_4th_bdry',
                      # 'glide_5th_bdry',
                      # 'glide_6th_bdry',
                      # 'glide_7th_bdry',
                      # 'glide_8th_bdry',
                      ),
               show_border=True,label='Closeness'),
        
        HGroup(Item('print_check',show_label=False),
                      Item('print_computation',show_label=False),
                      Item('print_error',show_label=False)),
       #----------------------------------------------------------------------
       show_border=False,label='GP1'),  
    #---------------------------------------------------------
    #---------------------------------------------------------
    Group(## 3rd-panel
         VGroup(HGroup('weight_fix'),
                HGroup(
                       'fix_all',
                       'fix_boundary',
                       'fix_boundary_i',
                       'fix_corner',),
                HGroup('boundary_z0',
                       'z0'),
                HGroup('selected_z0','selected_y0',
                       Item('fix_p_weight',show_label=False),),
                HGroup(Item('fix_button',show_label=False),
                       Item('unfix_button',show_label=False),
                       Item('clearfix_button',show_label=False)),  
             show_border=True,label='select'),          
          ###-------------------------------------
          HGroup('label',
                 Item('save_button',show_label=False),
                 label='Saving',show_border=True),
          ###-------------------------------------
       show_border=False,label='GP2'),                   
    #-----------------
    show_border=False,layout='tabbed'),  
    #---------------------------------------------------------
    #---------------------------------------------------------
    HGroup(Item('interactive',
                tooltip='InteractiveOptimization',),
           Item('_'),
           Item('optimize',show_label=False),
           Item('reinitialize',show_label=False),
           'hide_face','hide_edge',
           show_border=False),     
         #----------------    
         show_labels=False,show_border=False),                
    resizable=True,
    width = 0.04,
    )
    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------

    def __init__(self):
        GeolabComponent.__init__(self)

        self.optimizer = GP_DOINet()
        
        self.counter = 0
        
        self._fixed_vertex = []
        
        self.ref_glide_bdry_polyline = None

        self.snet_normal = self.snet_diagG_binormal = None
        
        self.data_opt_rectifystrip = None
    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------
    @property
    def mesh(self):
        return self.geolab.current_object.geometry
    
    @property
    def meshmanager(self):
        return self.geolab.current_object
    
    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------
    def geolab_settings(self):
        self.geolab.height = 600
        self.geolab.width = 600
        
    def object_open(self, file_name, geometry):
        name = ('mesh_{}').format(self.counter)
        self.geolab.add_object(geometry, name=name)
        self.counter += 1

    def object_change(self):
        pass

    def object_changed(self): #Huinote should not comment
        self.optimizer.mesh = self.geolab.current_object.geometry
        self.meshmanager.update_plot() # Hui add

    def object_save(self, file_name):
        self.optimizer.save_report(file_name)

    def set_state(self, state):
        if state != 'kr_interactive':
            self.interactive = False
        if state != 'mask_target':
            self.mask_target = False
     
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #                     GP-ALGORITHM: Common used:
    # -------------------------------------------------------------------------
    @on_trait_change('hide_face')
    def plot_hide_faces(self):
        if self.hide_face:
            self.meshmanager.hide_faces()
        else:
            self.meshmanager.plot_faces(color='white',#(100, 193, 151),#'grammarly',#,,
                                        glossy=1,
                                        opacity=1,
                                        )#color=(192,174,136)turquoise

    @on_trait_change('hide_edge')
    def plot_hide_edges(self):
        if self.hide_edge:
            self.meshmanager.hide_edges()
        else:
            self.meshmanager.plot_edges(color=(157,157,157),##(77,83,87),##
                                        tube_radius=0.4*self.meshmanager.r)
            #geo:red:(240,114,114);asy:blue:(98,113,180)
            #self.meshmanager.plot_edges(color =(98,113,180),tube_radius=1.2*self.meshmanager.r)

    @on_trait_change('fix_all')
    def plot_and_fix_all_vertices(self):
        name = 'fixall'
        v = np.arange(self.mesh.V)
        #v = self.mesh.patch_matrix[:-2,:-2].flatten()
        r = self.meshmanager.r
        if self.fix_all:
            self.mesh.fix(v)
            V = self.mesh.vertices[v]
            self.meshmanager.plot_glyph(points=V,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)

    @on_trait_change('fix_boundary')
    def plot_and_fix_boundary(self):
        "fixed_vertices='boundary"
        name = 'fixb'
        v,B = self.mesh.get_all_boundary_vertices()
        r = self.meshmanager.r
        if self.fix_boundary:
            self.mesh.fix(v)
            self.meshmanager.plot_glyph(points=B,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)  
            
    @on_trait_change('fix_boundary_i')
    def plot_and_fix_boundary_i(self):
        name = 'fixbi'
        if len(self.meshmanager.selected_edges)!=0:
            e = self.meshmanager.selected_edges 
        else:
            print('\n Please selecte an edge first!')        
        v,B = self.mesh.get_i_boundary_vertices(e[0],by_corner2=True) # need to choose
        r = self.meshmanager.r
        if self.fix_boundary_i:
            self.mesh.fix(v)
            self.meshmanager.plot_glyph(points=B,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)   

    @on_trait_change('fix_corner')
    def plot_and_fix_corner(self):
        name = 'fixc'
        v = self.mesh.corner #get_all_corner_vertices()
        C = self.mesh.vertices[v]
        if self.fix_corner:
            self.mesh.fix(v)
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)

    @on_trait_change('fix_button')
    def fix_current_handler_selected_vertex(self):
        name = 'fixv'
        v =  self.meshmanager.selected_vertices
        print(v)
        #v = np.array([14, 15, 16, 572, 573, 574])
        self._fixed_vertex.extend(v)
        self.mesh.fix(self._fixed_vertex)
        C = self.mesh.vertices[self._fixed_vertex]
        r = self.meshmanager.r
        self.meshmanager.plot_glyph(points=C,radius=2.5*r,
                                    shading = True,glossy=1,
                                    color=(255,144,64), ##red(210,92,106) #geo:red:(240,114,114);asy:blue:(98,113,180)
                                    name=name)#(210,92,106) #()(98,113,180)

    @on_trait_change('unfix_button')
    def unfix_current_handler_selected_vertex(self):
        name = 'fixv'
        v = self.meshmanager.selected_vertices
        self._fixed_vertex.remove(v[0])
        self.mesh.unfix(v)
        #self.meshmanager.remove(name)
        C = self.mesh.vertices[self._fixed_vertex]
        r = self.meshmanager.r
        self.meshmanager.plot_glyph(points=C,radius=2*r,color='r',name=name)

    @on_trait_change('clearfix_button')
    def clearfix_current_handler_selected_vertex(self):
        name = 'fixv'
        self.mesh.unfix(self._fixed_vertex)
        self._fixed_vertex = []
        self.meshmanager.remove(name)

    @on_trait_change('boundary_z0')
    def plot_and_bdry_z0(self):
        name = 'bdpt'
        v = np.array([119,68],dtype=int)
        C = self.mesh.vertices[v]
        if self.boundary_z0:
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,radius=2*r,
                                        shading = True,glossy=1,
                                        color='g',name=name)
        else:
            self.meshmanager.remove(name)


    @on_trait_change('selected_z0')
    def set_selected_vertices_xy_plane(self):
        name='s_z0'
        if self.selected_z0:
            self.optimizer.set_weight('selected_z0', 0.005)
            #print(self.meshmanager.selected_vertices)
            #ind = self.meshmanager.selected_vertices
            ind = np.array([985, 341, 426, 846, 629, 702, 264])
            self.optimizer.selected_v = ind
            vv = self.mesh.vertices[ind]
            self.meshmanager.plot_glyph(points=vv,glossy=1,
                                        radius=2*self.meshmanager.r,
                                        color = 'yellow',name=name)   
        else:
            self.meshmanager.remove(name)
            self.optimizer.set_weight('selected_z0', 0)
       
    @on_trait_change('selected_y0')
    def set_selected_vertices_xz_plane(self):
        name='s_y0'
        if self.selected_y0:
            self.optimizer.set_weight('selected_y0', 0.005)
            self.optimizer.selected_v = self.meshmanager.selected_vertices
            vv = self.mesh.vertices[self.meshmanager.selected_vertices]
            self.meshmanager.plot_glyph(points=vv,glossy=1,
                                        radius=2*self.meshmanager.r,
                                        color = 'yellow',name=name)   
        else:
            self.meshmanager.remove(name)
            self.optimizer.set_weight('selected_y0', 0)
  
        # ---------------------------------------------------------------------
        #                     Fairness weights:
        # ---------------------------------------------------------------------
    @on_trait_change('fair1')
    def set_fairness_1(self):
        self.mesh_fairness = self.boundary_fairness = 0.1
        self.tangential_fairness = self.spring_fairness = 0.1
    @on_trait_change('fair01')
    def set_fairness_01(self):
        self.mesh_fairness = self.boundary_fairness = 0.01
        self.corner_fairness = 0.01
        self.tangential_fairness = self.spring_fairness = 0.01
    @on_trait_change('fair001')
    def set_fairness_001(self):
        self.mesh_fairness = self.boundary_fairness = 0.005
        self.corner_fairness = 0.008
        self.tangential_fairness = self.spring_fairness = 0.0001
        self.fairness_diagmesh = 0.005
    @on_trait_change('fair0001')
    def set_fairness_0001(self):
        self.mesh_fairness = self.boundary_fairness = 0.0005
        self.corner_fairness = 0.0008
        self.tangential_fairness = self.spring_fairness = 0.0001
        self.fairness_diagmesh = 0.0005
    @on_trait_change('fair0')
    def set_fairness_0(self):
        self.mesh_fairness = self.boundary_fairness = 0
        self.corner_fairness = 0
        self.tangential_fairness = self.spring_fairness =  0
        self.fairness_diagmesh = 0
        self.fairness_4diff = 0
        self.fairness_diag_4diff = 0

    @on_trait_change('fairness_4diff')
    def set_fairness_4differential(self):
        if self.fairness_4diff:
            self.mesh_fairness = self.boundary_fairness = 0
            self.fairness_diagmesh = 0  
        #self.corner_fairness = 0
        #self.tangential_fairness = self.spring_fairness =  0

    @on_trait_change('close5')
    def set_close_5(self):
        self.reference_closeness = 0.5
        self.self_closeness = 0.5
    @on_trait_change('close1')
    def set_close_1(self):
        self.reference_closeness = 0.1
        self.self_closeness = 0.1
    @on_trait_change('close05')
    def set_close_05(self):
        self.reference_closeness = 0.05
        self.self_closeness = 0.05
    @on_trait_change('close01')
    def set_close_01(self):
        self.reference_closeness = 0.01
        self.self_closeness = 0.01
    @on_trait_change('close0')
    def set_close_0(self):
        self.reference_closeness = 0
        self.self_closeness = 0

    # -------------------------------------------------------------------------
    #                      Change topology (partial net/web)
    # ------------------------------------------------------------------------- 
    @on_trait_change('set_refer_mesh')
    def set_refermesh(self):
        if self.set_refer_mesh:
            if self.geolab.last_object is not None:
                #self.geolab.last_object.geometry.set_reference()
                self.mesh.reference_mesh = self.geolab.last_object.geometry
        else:
            self.mesh.reference_mesh = self.mesh.copy_mesh()#self.mesh
        rm = self.mesh.reference_mesh
        "get the index of boundary vertices:"
        ind = rm.boundary_curves(corner_split=False)[0]    
        "refine the boundary vertices"
        Ver = rm.vertices[ind]
        poly = Polyline(Ver,closed=True)  
        N = 5
        poly.refine(N)
        self.ref_glide_bdry_polyline = poly
        self.optimizer.glide_reference_polyline = self.ref_glide_bdry_polyline
        
    @on_trait_change('show_refer_mesh')
    def plot_reference_mesh(self):
        self.geolab.last_object.hide_faces()
        self.geolab.last_object.hide_edges()
        name = 'refer_mesh'
        if self.show_refer_mesh:
            try:
                rm = self.mesh.reference_mesh
            except:
                self.set_refermesh()
                rm = self.mesh.reference_mesh
            showe = Edges(rm,color ='black',tube_radius=0.5*self.meshmanager.r,
                          #color=(157,157,157),tube_radius=0.3*self.meshmanager.r,
                          name=name+'e')
            # showf = Faces(rm,color = (77,77,77),
            #               opacity=0.1,
            #               name=name+'f')
            self.meshmanager.add([showe])
        else:
            self.meshmanager.remove(name+'e')
            self.meshmanager.remove(name+'f')
            
    @on_trait_change('show_ref_mesh_boundary')
    def plot_reference_mesh_boundary(self):
        name = 'ref_mesh_pl'
        if self.show_ref_mesh_boundary:
            self.set_refermesh()
            rm = self.mesh.reference_mesh
            if True:
                poly = self.ref_glide_bdry_polyline
            else:
                poly = rm.boundary_polylines() ##Note: has problem!
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=0.7*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)

    def update_ith_boundary(self,N=3):
        self.optimizer.i_glide_bdry_crv = []
        self.optimizer.i_glide_bdry_ver = []
        if self.glide_1st_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=0)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_2nd_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=1)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_3rd_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=2)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_4th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=3)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)            
        if self.glide_5th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=4)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_6th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=5)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_7th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=6)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_8th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=7)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)               
    @on_trait_change('glide_1st_bdry')
    def plot_1st_boundary(self):
        name = '1stB'  
        if self.glide_1st_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=0)
            print(v)
            x = self.mesh.vertices[v][:,0]
            y = self.mesh.vertices[v][:,1]
            z = self.mesh.vertices[v][:,2]
            print('x:',np.mean(x),np.min(x),np.max(x))
            print('y:',np.mean(y),np.min(y),np.max(y))
            print('z:',np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
            ##print(self.mesh.boundary_curves(True),self.mesh.get_a_closed_boundary())
        else:
            self.meshmanager.remove(name)
      
    @on_trait_change('glide_2nd_bdry')
    def plot_2nd_boundary(self):
        name = '2ndB' 
        if self.glide_2nd_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=1)
            print(v)
            z = self.mesh.vertices[v][:,2]
            print(np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)

    @on_trait_change('glide_3rd_bdry')
    def plot_3rd_boundary(self):
        name = '3rdB'   
        if self.glide_3rd_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=2)
            print(v)
            z = self.mesh.vertices[v][:,2]
            print(np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_4th_bdry')
    def plot_4th_boundary(self):
        name = '4thB'  
        if self.glide_4th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=3)
            print(v)
            x = self.mesh.vertices[v][:,0]
            y = self.mesh.vertices[v][:,1]
            print('4th-x:',np.mean(x),np.min(x),np.max(x))
            print('4th-y:',np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_5th_bdry')
    def plot_5th_boundary(self):
        name = '5thB'  
        if self.glide_5th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=4)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_6th_bdry')
    def plot_6th_boundary(self):
        name = '6thB'  
        if self.glide_6th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=5)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_7th_bdry')
    def plot_7th_boundary(self):
        name = '7thB'  
        if self.glide_7th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=6)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)   
    @on_trait_change('glide_8th_bdry')
    def plot_8th_boundary(self):
        name = '8thB'  
        if self.glide_8th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=7)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)  
            
    @on_trait_change('show_corner')
    def showAllCorners(self):
        name = 'show_corner'
        if self.show_corner:
            C = self.mesh.vertices[self.mesh.corner]
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,color='r',
                                        radius=2*r,name=name)
        else:
            self.meshmanager.remove(name)

            
    # -------------------------------------------------------------------------
    #                    Weights + Plotting
    # -------------------------------------------------------------------------
    @on_trait_change('button_clear_constraint')
    def set_clear_webs(self):
        self.orthogonal = False
        self.GPC_net = False
        
        self.DOI_net = False
        self.is_DOI_SIR = False
        self.is_DOI_SIR_diagKite = False
        
        self.Kite_net = False
        self.is_Kite_diagGPC = False
        self.is_Kite_diagGPC_SIR = False
        
        self.CGC_net = False
        self.Gnet = False
          
        self.CNC_net = False
        self.Snet = False
        self.Snet_constR = False
        self.if_uniqR = False
        
        self.Anet = False
        
        self.Pseudogeodesic_net = False



    @on_trait_change('orient_rrv_normal')
    def set_orient_normals(self):
        if self.orient_rrv_normal:
            self.optimizer.orient_rrv_normal = True
            if self.oscu_rrv_tangent:
                self.optimizer.oscu_rrv_tangent = True
            else:
                self.optimizer.unit_edge_vec = True
        else:
            self.optimizer.orient_rrv_normal = False
            self.optimizer.oscu_rrv_tangent = False
            self.optimizer.unit_edge_vec = False
            
    @on_trait_change('CGC_net')
    def set_CGC_net(self): 
        if self.CGC_net:
            self.oscu_rrv_tangent = True
            self.orient_rrv_normal = True
        else:
            self.oscu_rrv_tangent = False
            self.orient_rrv_normal = False
            
    @on_trait_change('CNC_net')
    def set_CNC_net(self): 
        if self.CNC_net:
            self.Snet = True
            self.Snet_orient = True
            self.Snet_constR = True
        else:
            self.Snet = False
            self.Snet_orient = False
            self.Snet_constR = False

            
    @on_trait_change('Pseudogeodesic_net')
    def set_Pnet(self): 
        if self.Pseudogeodesic_net:
            self.orient_rrv_normal = True
            self.is_assigned_angle = True ##when it's true, the initial optimization won't explode up. so better turn it on alwayss
            #self.pseudogeo_orient = True  
            #self.pseudogeo_allSameAngle = True ##default is True
        else:
            self.orient_rrv_normal = False
            #self.pseudogeo_orient = False
            
    @on_trait_change('button_minimal_mesh')
    def set_orthogonal_Anet(self): 
        self.orthogonal = True  
        self.Anet = True

    @on_trait_change('button_CMC_mesh')
    def set_CMC_mesh(self): 
        self.orthogonal = True  
        self.Snet = True
        self.Snet_orient = True
        self.Snet_constR = True


    @on_trait_change('button_align_Kite')
    def set_align_ctrlKite(self): 
        if self.switch_diag_or_ctrl:
            self.DOI_net = True
            self.is_DOI_SIR = True
            self.is_DOI_SIR_diagKite = True
        else:
            self.Kite_net = True
            self.is_Kite_diagGPC = True
            self.is_Kite_diagGPC_SIR = True

    @on_trait_change('button_align_CGC')
    def set_align_CGC(self): 
        "(diagonal/control) Kite-net of the SIR-net is a CGC-net"
        self.set_align_ctrlKite()
        self.CGC_net = True

    @on_trait_change('button_align_CNC')
    def set_align_CNC(self): 
        "(diagonal/control) Kite-net of the SIR-net is a CNC-net"
        self.set_align_ctrlKite()
        self.CNC_net = True

    @on_trait_change('button_align_Pnet')
    def set_align_Pnet(self): 
        "(diagonal/control) Kite-net of the SIR-net is a P-net"
        self.set_align_ctrlKite()
        self.Pseudogeodesic_net = True

       
    #---------------------------------------------------------        
    #                       Ploting
    #---------------------------------------------------------     

    @on_trait_change('show_orient_vn')
    def plot_gonet_pseudogeodesic_vertexNormal(self):
        name = 'ps-geo-vn'
        if self.show_orient_vn:
            an,vn,_ = self.optimizer.get_orient_rr_normal()
            self.meshmanager.plot_vectors(anchor=an,vectors=vn,position='tail',
                                          color = 'g',name = name) 
        else:
            self.meshmanager.remove([name])    
            
    @on_trait_change('show_isoline')
    def plot_1st_or_2nd_isoline(self):
        name = 'ps-geo'
        if self.show_isoline:
            #work well, but without bdry-poly
            pl1,pl2,_,_,_,_ = self.mesh.get_rregular_split_list_for_polysegmnet(is_diagnet=False,is_poly=True)
            #pl1,pl2 = get_plot_ordered_isolines(self.mesh)
            
            if self.is_both:
                self.meshmanager.plot_polyline(pl1,color=(240,114,114),
                                               tube_radius=1.7*self.meshmanager.r,
                                               name=name+'1')
                self.meshmanager.plot_polyline(pl2,color=(98,113,180),
                                               tube_radius=1.7*self.meshmanager.r,
                                               name=name+'2')
            else:
                if self.switch_1st_or_2nd:
                    pl, clr, name = pl1, (240,114,114), name+'1'
                else:
                    pl, clr, name = pl2, (98,113,180), name+'2'
                
                self.meshmanager.plot_polyline(pl,color=clr,
                                               tube_radius=1.7*self.meshmanager.r,
                                               name=name)
        else:
            self.meshmanager.remove([name+'1',name+'2'])
            
    @on_trait_change('show_midpoint_edge1,show_midpoint_edge2')
    def plot_midpoint_edges(self):
        "same was as plot_isogonal_facebased_cross_vector; but different visulization"
        name = 'mid_edge'
        if self.show_midpoint_edge1 or self.show_midpoint_edge2:
            pl1,pl2 = self.mesh.get_quad_midpoint_cross_vectors(plot=True)
            if self.show_midpoint_edge1:
                self.meshmanager.plot_polyline(pl1,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(162,20,47),glossy=0.8,
                                               name = name+'1')
            else:
                self.meshmanager.remove([name+'1'])  
            if self.show_midpoint_edge2 and pl2 is not None:
                self.meshmanager.plot_polyline(pl2,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'2')
            else:
                self.meshmanager.remove([name+'2'])                                         
        else:
            self.meshmanager.remove([name+'1',name+'2'])      

    @on_trait_change('show_midpoint_polyline1,show_midpoint_polyline2')
    def plot_midpoint_polylines(self):
        name = 'mid_pl'
        if self.show_midpoint_polyline1 or self.show_midpoint_polyline2:
            pl1,pl2 = self.mesh.get_quad_midline()
            if self.show_midpoint_polyline1:
                self.meshmanager.plot_polyline(pl1,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(162,20,47),glossy=0.8,
                                               name = name+'1')
            else:
                self.meshmanager.remove([name+'1'])  
            if self.show_midpoint_polyline2 and pl2 is not None:
                self.meshmanager.plot_polyline(pl2,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'2')
            else:
                self.meshmanager.remove([name+'2'])                                         
        else:
            self.meshmanager.remove([name+'1',name+'2'])     
            
    @on_trait_change('show_midline_mesh')
    def plot_midline_mesh_checkboard(self):
        name = 'mid_mesh'
        if self.show_midline_mesh:
            dm = self.mesh.get_midline_mesh()
            self.save_new_mesh = dm
            self.label = name
            showe = Edges(dm,color=(0,59,117),
                          tube_radius=0.6*self.meshmanager.r,
                          glossy=0.8,
                          name=name+'e')
            self.meshmanager.add([showe])
        else:
            self.meshmanager.remove([name+'e',name+'f'])   
            
    @on_trait_change('show_diagonal_red_mesh')
    def plot_diagonal_red_mesh(self):
        name = 'dmesh1'
        if self.show_diagonal_red_mesh:
            dm = self.mesh.get_diagonal_mesh()
            ### geo:red:(240,114,114);asy:blue:(98,113,180)
            showe = Edges(dm,color ='white',#(98,113,180)(255,85,127),(0,59,117)
                          tube_radius=1*self.meshmanager.r, # 2*
                          glossy=1,
                          name=name+'e')
            # showf = Faces(dm,color ='white',#(100,0,0),
            #               opacity=0.8,
            #               name=name+'f')
            
            self.meshmanager.add([showe])
            self.save_new_mesh = dm
            self.label = name
        else:
            self.meshmanager.remove([name+'e',name+'f'])

    @on_trait_change('show_diagonal_blue_mesh')
    def plot_diagonal_blue_mesh(self):
        name = 'dmesh2'
        if self.show_diagonal_blue_mesh:
            dm = self.mesh.get_diagonal_mesh(blue=True)
            showe = Edges(dm,color =(0,59,117),#(0,0,100),#(85,85,225),
                          tube_radius=0.6*self.meshmanager.r,
                          glossy=0.8,
                          name=name+'e')
            # showf = Faces(dm,color ='white',#opacity=0.8,#(0,0,100),
            #               name=name+'f')
            self.meshmanager.add([showe])
            self.save_new_mesh = dm
            self.label = name
        else:
            self.meshmanager.remove([name+'e',name+'f'])


    @on_trait_change('show_diagonal_mesh')
    def plot_diagonal_mesh_checkboard(self):
        name = 'dmesh'
        if self.show_diagonal_mesh:
            dm = self.mesh.get_diagonal_mesh(whole=True) # whole mesh
            self.save_new_mesh = dm
            self.label = name
            showe = Edges(dm,color =(0,59,117), # red: a2142f 162,20,47; blue:(0,59,117))
                          tube_radius=0.6*self.meshmanager.r,
                          glossy=0.8,
                          name=name+'e')
            #showf = Faces(dm,color =(0,0,100),name=name+'f')
            self.meshmanager.add([showe])
        else:
            self.meshmanager.remove([name+'e',name+'f'])               


    @on_trait_change('show_revolution')
    def plot_rotational_mesh_from_a_patch(self):
        name = 'rot'
        if self.show_revolution:
            pi,rm = self.mesh.get_revolution(self.switch_GO_or_OG)    
            showe = Edges(rm,color ='black',name=name+'e')
            #showf = Faces(rm,color ='white',name=name+'f')
            ck,err = self.mesh.get_distortion_from_revolution(is_length=True,is_angle=False)
            showf = Faces(ck,face_data=err,glossy=1,opacity=0.7,
                          color='bwr',lut_range='-:0:+',name=name) 
            self.meshmanager.add([showe,showf])
            self.save_new_mesh = rm
            self.label = name
        else:
            self.meshmanager.remove([name+'e',name+'f'])

    #---------------------------------------------------------        
    #               CGC_net / Gnet - Ploting
    #---------------------------------------------------------

    @on_trait_change('show_oscu_tangent')
    def plot_osculating_tangent(self):
        name = 'oscut'
        if self.show_oscu_tangent:  
            an,t1,t2 = self.optimizer.get_osculating_tangents()
            self.meshmanager.plot_vectors(anchor=an,vectors=t1,position='tail',
                                          color = 'r',name = name+'1') 
            self.meshmanager.plot_vectors(anchor=an,vectors=t2,position='tail',
                                          color = 'b',name = name+'2') 
        else:
            self.meshmanager.remove([name+'1',name+'2'])

    @on_trait_change('show_cgc_centers')
    def plot_cgc_centers(self):
        name = 'cgc_c'
        if self.show_cgc_centers:  
            Cg1,Cg2,rho = self.optimizer.get_geodesic_curvature(self.switch_diag_or_ctrl)
            V = self.mesh.vertices[self.mesh.ver_rrv4f4]
            
            from archgeolab.archgeometry.curves import make_polyline_from_endpoints
            pl1 = make_polyline_from_endpoints(V,Cg1)
            pl2 = make_polyline_from_endpoints(V,Cg2)
            
            self.meshmanager.plot_polyline(pl1,color='r',name=name+'1')
            self.meshmanager.plot_polyline(pl2,color='b',name=name+'2')
        else:
            self.meshmanager.remove([name+'1',name+'2'])
            
    #---------------------------------------------------------        
    #               CNC_net / Anet / Snet - Ploting:
    #---------------------------------------------------------
    @on_trait_change('show_snet_center')
    def plot_snet_center(self):
        name = 'snetc'
        if self.show_snet_center:
            C,data = self.optimizer.get_snet_data(self.switch_diag_or_ctrl,center=True)
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,vertex_data=data,
                                        color='blue-red',lut_range='0:+',
                                        radius=2*r,name=name)
        else:
            self.meshmanager.remove(name)

    @on_trait_change('show_snet_normal')
    def plot_snet_normal(self):
        name = 'snetn'
        if self.show_snet_normal:
            an,N = self.optimizer.get_snet_data(self.switch_diag_or_ctrl,normal=True)
            self.meshmanager.plot_vectors(anchor=an,vectors=N,#normal neg or pos
                                          position='tail',color=(162,20,47),
                                          name=name)
            
            N0 = self.mesh.vertex_normals()
            N0[self.mesh.ver_rrv4f4] = N
            self.snet_normal = N0
        else:
            self.meshmanager.remove(name)
            
    @on_trait_change('show_snet_tangent')
    def plot_snet_tangent(self):
        name = 'snett'
        if self.show_snet_tangent:
            an,t1,t2 = self.optimizer.get_snet_data(self.switch_diag_or_ctrl,tangent=True)

            self.meshmanager.plot_vectors(anchor=an,vectors=t1,position='center',
                                          glyph_type = 'line',color='black',
                                          name=name+'1')
            self.meshmanager.plot_vectors(anchor=an,vectors=t2,position='center',
                                          glyph_type = 'line',color='black',
                                          name=name+'2')
        else:
            self.meshmanager.remove([name+'1',name+'2'])  
    
    @on_trait_change('show_vs_sphere')
    def plot_vertex_star_sphere(self):
        "S-net: vs-common-sphere"
        name='vs_sphere'
        v = self.meshmanager.selected_vertices
        if len(v) !=0:
            if self.show_vs_sphere:
                vv = np.array(v)
                C,r,Vneib = get_vs_interpolated_sphere(self.mesh.vertices,vv,self.mesh.ringlist)
                self.meshmanager.plot_glyph(points=Vneib,color=(123,123,0),
                                            name=name+'vi') 
                self.meshmanager.plot_glyph(points=C,color='black',
                                            name=name+'c')
                s = get_sphere_packing(C,r,Fa=50,Fv=50)
                shows = Faces(s,color = 'gray_40',opacity=0.3,name=name)
                self.meshmanager.add(shows)
            else:
                self.meshmanager.remove([name,name+'vi',name+'c'])
        else:
            print('Select a vertex first.')     
            

    #---------------------------------------------------------        
    #               Pnet - Ploting
    #---------------------------------------------------------
    @on_trait_change('show_pseudogeo_binormal')
    def plot_1st_or_2nd_osculatingNormal(self):
        name = 'ps-geo-n'
        if self.show_pseudogeo_binormal:
            an,on1,on2,cs13,cs24 = self.optimizer.pseudogeodesic_binormal(
                                            is_diagnet=self.switch_diag_or_ctrl,
                                            is_orientT=self.is_orient_tangent,
                                            is_orientN=self.is_orient_normal,
                                            is_remedy=self.is_remedied_BiN, 
                                            is_smooth=self.is_smoothed_BiN)
            self.optimizer.data_pseudogeodesic_binormal = [an,on1,on2,cs13,cs24]
            
            if self.switch_diag_or_ctrl:
                crvlists1,crvlists2 = self.mesh.all_rr_continuous_diag_polylist
            else:
                crvlists1,crvlists2 = self.mesh.all_rr_continuous_polylist##continue_family_poly

            if self.is_both:
                on, clr, name1 = on1, 'r', name+'1'
                self.meshmanager.plot_vectors(anchor=an,vectors=on,position='tail',
                                              color = clr,name = name1) 
                
                on, clr, name2 = on2, 'b', name+'2'
                self.meshmanager.plot_vectors(anchor=an,vectors=on,position='tail',
                                              color = clr,name = name2) 
            else:
                if self.switch_1st_or_2nd:
                    on, clr, name = on1, 'r', name+'1'
                    self.data_opt_rectifystrip = [crvlists1,an,on1]
                else:
                    on, clr, name = on2, 'b', name+'2'
                    self.data_opt_rectifystrip = [crvlists2,an,on2]
                self.meshmanager.plot_vectors(anchor=an,vectors=on,position='tail',
                                              color = clr,name = name) 
                
        else:
            self.meshmanager.remove([name+'1',name+'2'])   
            
    @on_trait_change('show_CGC_strip,show_CNC_strip,show_Pnet_rectifystrip')
    def plot_1st_or_2nd_isoline_strip(self):
        name = 'Isoline_strip'
        if self.show_CGC_strip or self.show_CNC_strip or self.show_Pnet_rectifystrip:
            width = self.strip_width * self.mesh.mean_edge_length() * 0.5
            
            if self.show_CGC_strip:
                sm1,_,sm2,_ = self.optimizer.get_CGC_circular_strip(
                    width,self.switch_diag_or_ctrl,self.is_central_strip)
            elif self.show_CNC_strip:
                sm1,_,sm2,_ = self.optimizer.get_CNC_circular_strip(
                    width,self.switch_diag_or_ctrl,self.is_central_strip)
            elif self.show_Pnet_rectifystrip:
                #if self.read_vertex_normal:
                    #biN =self.anvn[1]
                #else:
                biN = None
                sm1,_,sm2,_ = self.optimizer.pseudogeodesic_rectifying_srf(
                    width,biN,self.switch_diag_or_ctrl,self.is_central_strip)

            
            if self.is_both:
                sm, clr, name1 = sm1, 'orange', name+'1'
                data = sm.face_planarity()
                showf = Faces(sm,face_data=data,
                              lut_range=[0,np.max(data)],color='blue-red',#color=clr,
                              name=name1+'f') 
                showe = Edges(sm,color=clr,name=name1+'e')
                self.meshmanager.add([showf,showe])

                sm, clr, name2 = sm2, 'yellow', name+'2'
                data = sm.face_planarity()
                showf = Faces(sm,face_data=data,
                              lut_range=[0,np.max(data)],color='blue-red',#color=clr,
                              name=name2+'f') 
                showe = Edges(sm,color=clr,name=name2+'e')
                self.meshmanager.add([showf,showe])
                
            else:
                if self.switch_1st_or_2nd:
                    sm, clr, name = sm1, 'orange', name+'1'
                else:
                    sm, clr, name = sm2, 'yellow', name+'2'
                    
                data = sm.face_planarity()
                showf = Faces(sm,face_data=data,
                              lut_range=[0,np.max(data)],color='blue-red',#color=clr,
                              name=name+'f') 
                showe = Edges(sm,color=clr,name=name+'e')
                self.meshmanager.add([showf,showe])
                
                print('planarity:[min,mean,max]=','%.2g' % np.min(data),'%.2g' % np.mean(data),'%.2g' % np.max(data))
                self.save_new_mesh, self.label = sm, name
        else:
            self.meshmanager.remove([name+'1e',name+'1f',name+'2e',name+'2f'])   
            

    @on_trait_change('show_isolinestrip_unroll')
    def plot_1st_or_2nd_rectifyingDevelopableSrf_unrolling(self):
        from huilab.huimesh.unroll import unroll_multiple_strips
        dist = self.mesh.mean_edge_length() * 0.5#self.scale_dist_offset
        width = self.strip_width * dist
        name = 'ps-R2D'
        if self.show_isolinestrip_unroll:

            if self.show_CGC_strip:
                sm1,list1,sm2,list2 = self.optimizer.get_CGC_circular_strip(
                    width,self.switch_diag_or_ctrl,self.is_central_strip)
            elif self.show_CNC_strip:
                sm1,list1,sm2,list2 = self.optimizer.get_CNC_circular_strip(
                    width,self.switch_diag_or_ctrl,self.is_central_strip)
            elif self.show_Pnet_rectifystrip:
                sm1,list1,sm2,list2 = self.optimizer.pseudogeodesic_rectifying_srf(
                    width,None,self.switch_diag_or_ctrl,self.is_central_strip)
            else:
                print('!!!--Need to choose CGCstrip, CNCstrip or Pstrip!!!')
                sys.exit()            
            

            if self.is_both:
                sm,lists, clr, name1 = sm1,list1, (240,114,114), name+'1'
                um = unroll_multiple_strips(sm,lists,dist,
                                            step=self.dist_inverval,coo=2,
                                            anchor=0,efair=self.set_unroll_strip_fairness,
                                            itera=100,
                                            is_midaxis=self.is_unroll_midaxis,
                                            w_straight=self.set_unroll_strip_fairness)
                showf = Faces(um,color=clr,name=name1+'f') 
                showe = Edges(um,color=clr,name=name1+'e')
                self.meshmanager.add([showf,showe])
                
                sm,lists, clr, name2 = sm2,list2, (98,113,180), name+'2'
                um = unroll_multiple_strips(sm,lists,dist,
                                            step=self.dist_inverval,coo=2,
                                            anchor=0,efair=self.set_unroll_strip_fairness,
                                            itera=100,
                                            is_midaxis=self.is_unroll_midaxis,
                                            w_straight=self.set_unroll_strip_fairness)
                showf = Faces(um,color=clr,name=name2+'f') 
                showe = Edges(um,color=clr,name=name2+'e')
                self.meshmanager.add([showf,showe])

            else:
                if self.switch_1st_or_2nd:#geo:red:(240,114,114);asy:blue:(98,113,180)
                    sm,lists, clr, name = sm1,list1, (240,114,114), name+'1'
                else:
                    sm,lists, clr, name = sm2,list2, (98,113,180), name+'2'
                    
                um = unroll_multiple_strips(sm,lists,dist,
                                            step=self.dist_inverval,coo=2,
                                            anchor=0,efair=self.set_unroll_strip_fairness,
                                            itera=100,
                                            is_midaxis=self.is_unroll_midaxis,
                                            w_straight=self.set_unroll_strip_fairness)
                showf = Faces(um,color=clr,name=name+'f') 
                showe = Edges(um,color=clr,name=name+'e')
                self.meshmanager.add([showf,showe])
                self.save_new_mesh, self.label = um, name
        else:
            self.meshmanager.remove([name+'1e',name+'1f',name+'2e',name+'2f'])                 
 
    #--------------------------------------------------------------------------
    #                         Printing / Check
    #--------------------------------------------------------------------------         
    @on_trait_change('print_computation')
    def print_computation_info(self):
        print('No. of all vertices: ', self.mesh.V)
        print('No. of all faces: ', self.mesh.F)
        print('No. of all edges: ', self.mesh.E)
        
        print('No. of rr_vertices: ', self.mesh.num_rrv4f4)
        print('No. of rr_quad_faces: ', self.mesh.num_rrf) #circular mesh
        
        print('#variables: ', len(self.optimizer.X))
        print('#constraints: ', len(self.optimizer._r))
        #print('time[s] per iteration: ',  )

    @on_trait_change('print_check')
    def print_isogonal_data(self):
        #print('theta:[min,mean,max]=','%.3f'%angle_min,'%.3f'%angle_mean,'%.3f'%angle_max)
        print('check')

    @on_trait_change('print_error')
    def print_errors(self):
        self.optimizer.make_errors()
    #--------------------------------------------------------------------------
    #                                    Save txt / obj
    #--------------------------------------------------------------------------
    @on_trait_change('save_button')
    def save_file(self):
        name = ('{}').format(self.label)
        if self.save_new_mesh is None:
            self.save_new_mesh = self.mesh
            pass
        #save_path = '/objs'     
        
        save_path = r'/Users/wanghui/Github/DOI/objs'    
        completeName = os.path.join(save_path, name)
        self.save_new_mesh.make_obj_file(completeName)

        print('\n\n NOTE: <'+self.label+'> mesh has been saved in <'+completeName+'>\n')
    # -------------------------------------------------------------------------
    #                              Settings
    # -------------------------------------------------------------------------
    def set_settings(self):
        # ---------------------------------------------------------------------
        #                     GP-ALGORITHM common used:
        # ---------------------------------------------------------------------
        self.optimizer.threshold = 1e-20
        #self.optimizer.itera_run = self.itera_run
        #self.optimizer.iterations = self.itera_run
        self.optimizer.epsilon = self.epsilon
        self.optimizer.step = self.step
        self.optimizer.fairness_reduction = self.fairness_reduction
        
        self.optimizer.add_weight('mesh_fairness', self.mesh_fairness)
        self.optimizer.add_weight('tangential_fairness', self.tangential_fairness)
        self.optimizer.add_weight('boundary_fairness', self.boundary_fairness)
        self.optimizer.add_weight('corner_fairness', self.corner_fairness)
        self.optimizer.add_weight('spring_fairness', self.spring_fairness)
        self.optimizer.add_weight('fairness_4diff', self.fairness_4diff)
        self.optimizer.add_weight('fairness_diag_4diff', self.fairness_diag_4diff)
        self.optimizer.add_weight('fairness_diagmesh', self.fairness_diagmesh)
 
        self.optimizer.add_weight('reference_closeness', self.reference_closeness)
        self.optimizer.add_weight('self_closeness', self.self_closeness)
        
        self.optimizer.add_weight('boundary_glide', self.boundary_glide)
        self.optimizer.add_weight('i_boundary_glide', self.i_boundary_glide)
        self.optimizer.add_weight('fixed_vertices', self.weight_fix)
        self.optimizer.add_weight('fix_point',  self.fix_p_weight)
        self.optimizer.add_weight('fix_corners',  self.fix_corner)
        self.optimizer.set_weight('sharp_corner',  self.sharp_corner) 
        self.optimizer.set_weight('z0', self.z0)
        self.optimizer.set_weight('boundary_z0', self.boundary_z0)
        
        # ---------------------------------------------------------------------
        self.optimizer.oscu_rrv_tangent = self.oscu_rrv_tangent
        self.optimizer.orient_rrv_normal = self.orient_rrv_normal
        
        self.optimizer.set_weight('orthogonal',  self.orthogonal*1)
        
        self.optimizer.set_weight('DOI', self.DOI_net)
        self.optimizer.is_DOI_SIR = self.is_DOI_SIR
        self.optimizer.is_DOI_SIR_diagKite = self.is_DOI_SIR_diagKite
        
        self.optimizer.set_weight('Kite', self.Kite_net)
        self.optimizer.is_Kite_diagGPC = self.is_Kite_diagGPC
        self.optimizer.is_Kite_diagGPC_SIR = self.is_Kite_diagGPC_SIR
        
        self.optimizer.set_weight('Gnet', self.Gnet)
        self.optimizer.set_weight('CGC', self.CGC_net)
        
        self.optimizer.set_weight('Anet',  self.Anet)
        self.optimizer.set_weight('Snet', self.Snet)
        self.optimizer.set_weight('Snet_orient', self.Snet_orient)
        self.optimizer.set_weight('Snet_constR', self.Snet_constR)
        self.optimizer.if_uniqradius = self.if_uniqR
        self.optimizer.assigned_snet_radius = self.Snet_constR_assigned

        self.optimizer.set_weight('Pnet', self.Pseudogeodesic_net)
        self.optimizer.is_assigned_angle = self.is_assigned_angle 
        self.optimizer.assigned_angle = self.assigned_angle      
        
        self.optimizer.is_GO_or_OG = self.switch_GO_or_OG 
        self.optimizer.is_diag_or_ctrl = self.switch_diag_or_ctrl
        self.optimizer.is_Kite_switch = self.switch_kite_1_or_2

    # -------------------------------------------------------------------------
    #                         Reset + Optimization
    # -------------------------------------------------------------------------
    def reinitialize_constraints(self):# Hui: to set
        self.optimizer.is_initial = True 
        #self.planarity = False
        #self.orthogonal = False
        #self.isogonal = False

    @on_trait_change('reinitialize')
    def reinitialize_optimizer(self):
        self.reinitialize_constraints()
        self.set_settings()
        self.optimizer.reinitialize = True
        self.optimizer.initialization() # Hui add
        self.mesh.vertices = self.mesh.vertices_0 # Huinote:add
        self.meshmanager.update_plot()
        print('\n---------------------\n')

    def updating_plot(self):
        pass

    def optimization_step(self):
        if not self.interactive:
            self.handler.set_state(None)
        self.set_settings()
        self.optimizer.optimize()
        #self.print_error()
        self.updating_plot()
        self.meshmanager.update_plot()
        if self.fairness_reduction !=0:
            self.mesh_fairness = self.mesh_fairness/(10**(self.fairness_reduction))
            self.tangential_fairness = self.tangential_fairness/(10**
                                        (self.fairness_reduction))
            self.boundary_fairness = self.boundary_fairness/(10**
                                    (self.fairness_reduction))
            self.spring_fairness = self.spring_fairness/(10**
                                    (self.fairness_reduction))    

    @on_trait_change('optimize')
    def optimize_mesh(self):
        import time
        start_time = time.time()
        itera = self.itera_run
        self.meshmanager.iterate(self.optimization_step, itera) # note:iterations from gpbase.py
        self.meshmanager.update_plot()
        
        print('time[s] per iteration:','%.3g s' %((time.time() - start_time)/itera))
            
    @on_trait_change('interactive')
    def interactive_optimize_mesh(self):
        self.handler.set_state('kr_interactive')
        if self.interactive:
            def start():
                self.mesh.handle = self.meshmanager.selected_vertices
            def interact():
                self.meshmanager.iterate(self.optimization_step,1)
            def end():
                self.meshmanager.iterate(self.optimization_step,5)
            self.meshmanager.move_vertices(interact,start,end)
        else:
            self.mesh.handle = None
            self.meshmanager.move_vertices_off()
