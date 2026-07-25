# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:13:42 2025

@author: wanghui
"""

# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact hwangchn@outlook.com

#------------------------------------------------------------------------------
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


import os
# 禁用Qt6相关后端
os.environ["QT_API"] = "pyqt5"
os.environ["ETS_TOOLKIT"] = "qt"
# 屏蔽vtk/mayavi自动加载Qt6逻辑
os.environ["VTK_USE_QT6"] = "0"

import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path)

#print(path) ##/Users/X/Github/DOS
#------------------------------------------------------------------------------



# -------------------------------------------------------------------------
#                              Run
# -------------------------------------------------------------------------
if __name__ == '__main__':

    a = path + r'/objs'

    rot = a + r'/obj_rot'
    snet = a + r'/obj_snet'

    file = snet +r'/cmc1.obj'
    #file = rot +r'/conical1_diagKite.obj'
    #file = rot +r'/chebyshev_sphere.obj'
    
    file = rot +r'/rot2_unitscale_cut_CNC_seam_pciso2.obj'
    #file = rot +r'/unduloid_cut_unitscale_CNC_r=2_seam_pciso3.obj'
    #file = rot +r'/unduloid1_cusp_unitscale_CNC-r=0.25_seam_pciso3.obj'
    
    #file = rot +r'/crpc/rot_a_pos_50_crpc_uv_seam.obj'
    #file = rot +r'/crpc/rot_a_pos_1_crpc_uv_seam.obj'
    #file = rot +r'/crpc/rot_a_pos_5_crpc_uv_seam_pciso2.obj'
    
    #file = rot +r'/pseudosphere_unitscale_seam_pciso1.obj'
    # file = rot +r'/crpc/rot_a_minus_50_crpc_uv_seam.obj'
    # file = rot +r'/crpc/rot_a_minus_1_crpc_uv_seam.obj'
    # file = rot +r'/crpc/rot_a_minus_5_crpc_uv_seam_pciso1.obj'
    
    
    #file = rot +r'/rot2_unitscale_cut_CNC.obj'
    #file = rot +r'/unduloid1_2cusps.obj'
    #file = rot +r'/unduloid_cut_unitscale.obj'
    
    # file = rot +r'/pseudosphere_unitscale.obj'
    # reffile = rot +r'/pseudosphere_unitscale_sub2.obj'
    
    #file = a +r'/richmond_polar4_sparse.obj'
    
    #file =r'/Users/wanghui/Desktop/geometrylab7/obj_PQ/evolute_ex8.obj'
    #file =r'/Users/wanghui/Desktop/geometrylab7/obj_rotation/undoloid2_cut2_remesh.obj'

    #----------------------------------------

    '''Instantiate the sample component'''
    from doi__2_gui import DOINet
    component = DOINet()

    '''Instantiate the main geolab application'''
    from archgeolab.archgeometry.gui_basic import GeolabGUI
    GUI = GeolabGUI()

    '''Add the component to geolab'''
    GUI.add_component(component)
    
    '''Open an obj file'''
    GUI.open_obj_file(file)
    
    '''Open another obj file'''
   # GUI.open_obj_file(reffile)
    
    '''Start geolab main loop'''
    GUI.start()

