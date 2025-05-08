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
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    file = rot +r'/conical1_diagKite.obj'
    #file = rot +r'/chebyshev_sphere.obj'
    file = rot +r'/chebyshev_sphere_cut3.obj'
    
    #file =r'/Users/wanghui/Desktop/geometrylab7/obj_rotation/rot1.obj'
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
    #GUI.open_obj_file(reffile)
    
    '''Start geolab main loop'''
    GUI.start()

