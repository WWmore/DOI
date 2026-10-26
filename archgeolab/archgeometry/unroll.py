# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:00:39 2021

@author: wangh0m
"""

#---------------------------------------------------------------------------
import numpy as np
#import time
from scipy import sparse
from scipy.optimize import fsolve
from geometrylab.geometry.meshpy import Mesh
#---------------------------------------------------------------------------

def unroll_multiple_strips1(sm,num_list,dist=0,step=1.2,
                           coo=2,anchor=0,efair=0.005,
                           ee=0.001,itera=10,is_midaxis=False,
                           w_straight=0,
                           is_V_flip=False):
    """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
    Parameters
    ----------
    Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
    anchor : starting coordinate on x-axis for layout
    step : control the distance between different strips
    coo : reserved, dist≠0布局统一沿x轴排列
    dist:
        dist ==0:
            1.对每条strip：提取中线polyline（横截边中点连线）
            2.对该strip中线首尾两点连线做刚体旋转，**使得首尾连线的中垂线为x轴**
            3.求旋转后中线polyline与x轴交点(x0,0,0)
            4.使用平移向量(-x0,0,0)平移整个strip，交点落到原点
        dist !=0:
            1. 对每条strip：取首、中间、末尾横截边中点三点插值圆弧
            2. 所有圆弧按圆弧长度【由短到长】排序
            3. 将圆弧中点沿X轴依次排布，取消强制x轴对称
    """
    Vall = sm.vertices
    allv = np.empty((0,3), dtype=np.float64)
    allf = np.empty((0,4), dtype=int)

    istart_input = 0       # 原始输入Vall读取游标
    v_offset_out = 0        # 输出allv顶点偏移

    print("--------------------")
    # 先预展开所有strip，缓存数据
    strip_cache = []
    for num in num_list:
        Vi = Vall[istart_input:istart_input + 2*num]
        if is_V_flip:
            Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))
        _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
        istart_input += 2*num
        if vlist is not None:
            strip_cache.append({"num":num, "v":vlist, "f":flist})

    def find_polyline_intersect_xaxis(poly):
        """求polyline折线段和X轴【y=0,z=0】交点，返回 (x,0,0)"""
        eps = 1e-12
        for i in range(poly.shape[0]-1):
            pa = poly[i]
            pb = poly[i+1]
            dy = pb[1] - pa[1]
            dz = pb[2] - pa[2]
            t_candidates = []
            if abs(dy) > eps:
                t1 = -pa[1]/dy
                t_candidates.append(t1)
            if abs(dz) > eps:
                t2 = -pa[2]/dz
                t_candidates.append(t2)
            if len(t_candidates) == 0:
                continue
            t0 = t_candidates[0]
            valid = True
            for t in t_candidates[1:]:
                if not np.isclose(t, t0, atol=1e-10):
                    valid=False
                    break
            if not valid:
                continue
            if 0 <= t0 <= 1:
                pt = pa + t0*(pb-pa)
                return np.array([pt[0],0,0])
        # 找不到真实交点返回None
        return None

    print("1111111111111111111")
    # ================= dist=0分支 =================
    intersect_list = [] # 新增：收集所有strip交点
    for s in strip_cache:
        num = s["num"]
        vlist = s["v"]
        flist = s["f"]
        #1.提取原始中线polyline
        mid_line = 0.5 * (vlist[:num] + vlist[num:2*num])
        p_start = mid_line[0]
        p_end = mid_line[-1]

        def rotate_make_midperp_xaxis(pa, pb):
            mid = (pa + pb)/2.0
            pa_loc = pa - mid
            pb_loc = pb - mid
            seg_vec = pb_loc - pa_loc
            vec_len = np.linalg.norm(seg_vec)
            if vec_len < 1e-12:
                return np.eye(3), mid
            target_pa = np.array([0, pa_loc[1], pa_loc[2]])
            H = np.outer(pa_loc, target_pa)
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            return R, mid

        R, pm = rotate_make_midperp_xaxis(p_start, p_end)
        v_shifted = vlist - pm
        v_rot = (R @ v_shifted.T).T
        v_rot = v_rot + pm

        mid_rot = 0.5 * (v_rot[:num] + v_rot[num:2*num])

        intersect_pt = find_polyline_intersect_xaxis(mid_rot)
        if intersect_pt is not None:
            x0 = intersect_pt[0]
            delta_trans = np.array([-x0, 0.0, 0.0])
            pt_final = intersect_pt + delta_trans
        else:
            print(s, "warning: 中线polyline不与X轴相交")
            delta_trans = np.array([0.0,0.0,0.0])
            pt_final = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        print(intersect_pt, delta_trans)
        intersect_list.append(pt_final)
        print(s,intersect_list)
            
        v_final = v_rot + delta_trans
        flist_shifted = flist + v_offset_out
        allv = np.vstack((allv, v_final))
        allf = np.vstack((allf, flist_shifted))
        v_offset_out += len(v_final)

    print("22222222222222222222")
    
    # 保护：空网格保护
    if allv.shape[0]==0:
        print("WARNING: allv empty!")
        intersect_all = np.empty((0,3), dtype=np.float64)
        allsm = Mesh()
        return allsm, intersect_all
    
    allsm = Mesh()
    allsm.make_mesh(allv, allf)
    
    print("333333333333333333333")
    
    # 转为 N×3 np.array
    print("----------",intersect_list,"----------")
    intersect_all = np.array(intersect_list, dtype=np.float64)
    return allsm, intersect_all



##针对dist==0情况，之前可以运行，所有strip平移到了原点，但后来又不可行，不确定原因是什么，else pass行为表现奇怪
# def unroll_multiple_strips11(sm,num_list,dist=None,step=1.2,
#                            coo=2,anchor=0,efair=0.005,
#                            ee=0.001,itera=10,is_midaxis=False,
#                            w_straight=0,
#                            is_V_flip=False):
#     """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
#     Parameters
#     ----------
#     Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
#     anchor : starting coordinate on x-axis for layout
#     step : control the distance between different strips
#     coo : reserved, dist≠0布局统一沿x轴排列
#     dist:
#         dist ==0:
#             1.对每条strip：提取中线polyline（横截边中点连线）
#             2.平移strip：使中线polyline和X轴(y=0,z=0)交点为原点(0,0,0)
#             3.绕原点做刚体旋转，使得中线起点、终点连线的中垂线为X轴
#         dist !=0:
#             1. 对每条strip：取首、中间、末尾横截边中点三点插值圆弧
#             2. 所有圆弧按圆弧长度【由短到长】排序
#             3. 将圆弧中点沿X轴依次排布，取消强制x轴对称
#     """
#     Vall = sm.vertices
#     allv = np.empty((0,3), dtype=np.float64)
#     allf = np.empty((0,4), dtype=int)


#     istart_input = 0       # 原始输入Vall读取游标
#     v_offset_out = 0        # 输出allv顶点偏移
    
#     # 先预展开所有strip，缓存数据
#     strip_cache = []
#     for num in num_list:
#         Vi = Vall[istart_input:istart_input + 2*num]
#         if is_V_flip:
#             Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))
#         _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
#         istart_input += 2*num
#         if vlist is not None:
#             strip_cache.append({"num":num, "v":vlist, "f":flist})
    
#     def find_polyline_intersect_xaxis(poly):
#         """求polyline折线段和X轴【y=0,z=0】交点，返回 (x,0,0)"""
#         for i in range(poly.shape[0]-1):
#             pa = poly[i]
#             pb = poly[i+1]
#             dx = pb[0] - pa[0]
#             dy = pb[1] - pa[1]
#             dz = pb[2] - pa[2]
#             if abs(dy) < 1e-12 or abs(dz) <1e-12:
#                 continue
#             t1 = -pa[1]/dy
#             t2 = -pa[2]/dz
#             if not np.isclose(t1,t2,atol=1e-10):
#                 continue
#             t = t1
#             if 0.0 <= t <= 1.0:
#                 x_val = pa[0] + t*dx
#                 return np.array([x_val, 0.0, 0.0])
#         #兜底：若无真实交点，返回中线中点投影到X轴
#         mid_pt = poly[poly.shape[0]//2]
#         return np.array([mid_pt[0], 0.0, 0.0])
    
#     #if dist == 0:
#     # ================= dist=0分支 =================
#     for s in strip_cache:
#         num = s["num"]
#         vlist = s["v"]
#         flist = s["f"]
#         #1.提取原始中线polyline
#         mid_line = 0.5 * (vlist[:num] + vlist[num:2*num])
#         #2.求中线与X轴交点(x0,0,0)
#         intersect_pt = find_polyline_intersect_xaxis(mid_line)
#         x0 = intersect_pt[0]
#         #平移向量：沿X轴平移(-x0,0,0)，使交点变为(0,0,0)
#         delta_trans = np.array([-x0, 0.0, 0.0])
#         v_trans = vlist + delta_trans
#         #平移后的中线
#         mid_trans = 0.5 * (v_trans[:num] + v_trans[num:2*num])
#         p_start = mid_trans[0]
#         p_end = mid_trans[-1]

#         #3.绕原点旋转：起点终点连线的中垂线=X轴
#         #几何条件：p_start=(t, s, q), p_end=(t, -s, -q)，两点连线垂直X轴
#         def rotate_to_midperp_xaxis(pa, pb):
#             # pa: start, pb: end
#             vec_se = pb - pa
#             #目标向量：两点连线沿Y-Z平面，x分量=0
#             target_vec = np.array([0.0, vec_se[1], vec_se[2]])
#             H = np.outer(vec_se, target_vec)
#             U, S, Vt = np.linalg.svd(H)
#             R = Vt.T @ U.T
#             #修正反射
#             if np.linalg.det(R) < 0:
#                 Vt[-1, :] *= -1
#                 R = Vt.T @ U.T
#             return R

#         R = rotate_to_midperp_xaxis(p_start, p_end)
#         #旋转全部顶点，绕原点
#         v_final = (R @ v_trans.T).T

#         flist_shifted = flist + v_offset_out
#         allv = np.vstack((allv, v_final))
#         allf = np.vstack((allf, flist_shifted))
#         v_offset_out += len(v_final)
    
#     allsm = Mesh()
#     allsm.make_mesh(allv, allf)
#     return allsm


def unroll_multiple_strips2(sm,num_list,dist=0,step=1.2,
                           coo=2,anchor=0,efair=0.005,
                           ee=0.001,itera=10,is_midaxis=False,
                           w_straight=0,
                           is_V_flip=False):
    """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
    Parameters
    ----------
    Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
    anchor : starting coordinate on x-axis for layout
    step : control the distance between different strips
    coo : reserved, dist≠0布局统一沿x轴排列
    dist:
        dist ==0:
            1.对每条strip：提取中线polyline（横截边中点连线）
            2.平移strip：使中线polyline和X轴(y=0,z=0)交点为原点(0,0,0)
            3.绕原点做刚体旋转，使得中线起点、终点连线的中垂线为X轴
        dist !=0:
            1. 对每条strip：取首、中间、末尾横截边中点三点插值圆弧
            2. 所有圆弧按圆弧长度【由短到长】排序
            3. 将圆弧中点沿X轴依次排布，取消强制x轴对称
    """
    Vall = sm.vertices
    allv = np.empty((0,3), dtype=np.float64)
    allf = np.empty((0,4), dtype=int)

    istart_input = 0       # 原始输入Vall读取游标
    v_offset_out = 0        # 输出allv顶点偏移

    # 先预展开所有strip，缓存数据
    strip_cache = []
    for num in num_list:
        Vi = Vall[istart_input:istart_input + 2*num]
        if is_V_flip:
            Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))
        _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
        istart_input += 2*num
        if vlist is not None:
            strip_cache.append({"num":num, "v":vlist, "f":flist})
            
    ##if dist !=0:
    strip_info = []
    for s in strip_cache:
        num = s["num"]
        v = s["v"]
        #f = s["f"]
        mid_poly = 0.5 * (v[:num] + v[num:2*num])
        seg_len = np.linalg.norm(np.diff(mid_poly, axis=0), axis=1)
        poly_total_len = np.sum(seg_len)
        rot_center = mid_poly[num//2]
        p0_orig = mid_poly[0]
        pe_orig = mid_poly[-1]
        chord_mid_orig = 0.5 * (p0_orig + pe_orig)

        strip_info.append({
            "strip":s,
            "poly_len": poly_total_len,
            "mid_poly": mid_poly,
            "rot_center": rot_center,
            "chord_mid_orig": chord_mid_orig
        })
    strip_info.sort(key=lambda d:d["poly_len"])

    current_x = anchor
    for item in strip_info:
        s = item["strip"]
        num = s["num"]
        vlist = s["v"]
        flist = s["f"]
        mid_poly = item["mid_poly"]
        rot_center = item["rot_center"]

        v_local = vlist - rot_center
        poly_local = mid_poly - rot_center
        p0 = poly_local[0]
        pe = poly_local[-1]

        r0 = np.linalg.norm(p0)
        re = np.linalg.norm(pe)
        R = np.eye(3)
        if r0>1e-12 and re>1e-12:
            chord_len = np.linalg.norm(p0 - pe)
            half_chord = chord_len / 2.0
            p0_target = np.array([0, half_chord, 0.0])
            pe_target = np.array([0, -half_chord, 0.0])

            P = np.vstack([p0, pe]).T
            Q = np.vstack([p0_target, pe_target]).T
            H = P @ Q.T
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1,:] *= -1
                R = Vt.T @ U.T
            v_rot = (R @ v_local.T).T
            poly_rot = (R @ poly_local.T).T
        else:
            v_rot = v_local
            poly_rot = poly_local

        p0_rot = poly_rot[0]
        pe_rot = poly_rot[-1]
        chord_mid_rot = 0.5 * (p0_rot + pe_rot)
        target_chord_mid = np.array([current_x, 0.0, 0.0])
        delta = target_chord_mid - chord_mid_rot
        v_final = v_rot + delta

        flist_shifted = flist + v_offset_out
        allv = np.vstack((allv, v_final))
        allf = np.vstack((allf, flist_shifted))
        v_offset_out += len(v_final)
        current_x += step * dist

    allsm = Mesh()
    allsm.make_mesh(allv, allf)
    return allsm

# ##dist=0时候，重合正确，但是strip上下不对称。dist非零时，不正确
# def unroll_multiple_strips11(sm,num_list,dist=None,step=1.2,
#                            coo=2,anchor=0,efair=0.005,
#                            ee=0.001,itera=10,is_midaxis=False,
#                            w_straight=0,
#                            is_V_flip=False):
#     """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
#     Parameters
#     ----------
#     Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
#     anchor : starting coordinate on x-axis for layout
#     step : control the distance between different strips
#     coo : reserved, dist≠0布局统一沿x轴排列
#     dist:
#         dist ==0: 所有strip，按四边形数量奇偶取锚点，平移至坐标原点[0,0,0]
#             奇数四边形：取中间四边形面重心；偶数四边形：取中间横截边中点
#         dist !=0:
#             1. 对每条strip：取首、中间、末尾横截边中点三点插值圆弧
#             2. 所有圆弧按圆弧长度【由短到长】排序
#             3. 将圆弧中点沿X轴依次排布，取消强制x轴对称
#     """
#     Vall = sm.vertices
#     allv = np.empty((0,3), dtype=np.float64)
#     allf = np.empty((0,4), dtype=int)

#     istart_input = 0       # 原始输入Vall读取游标
#     v_offset_out = 0        # 输出allv顶点偏移

#     # 先预展开所有strip，缓存数据
#     strip_cache = []
#     for num in num_list:
#         Vi = Vall[istart_input:istart_input + 2*num]
#         if is_V_flip:
#             Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))
#         _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
#         istart_input += 2*num
#         if vlist is not None:
#             strip_cache.append({"num":num, "v":vlist, "f":flist})

#     if dist == 0:
#         # ================= dist=0分支：全部strip锚点平移到原点，不旋转 =================
#         for s in strip_cache:
#             num = s["num"]
#             vlist = s["v"]
#             flist = s["f"]
#             # 按奇偶规则求本strip原始锚点
#             if num % 2 == 1:
#                 # 奇数四边形：中间四边形面重心
#                 mid_quad_idx = num // 2
#                 quad_v_idx = flist[mid_quad_idx]
#                 quad_verts = vlist[quad_v_idx]
#                 anc_orig = np.mean(quad_verts, axis=0)
#             else:
#                 # 偶数四边形：中间横截边中点
#                 im = num // 2
#                 anc_orig = 0.5 * (vlist[im] + vlist[num + im])
#             # 整体平移，锚点移动到原点
#             delta = np.array([0.,0.,0.]) - anc_orig
#             v_trans = vlist + delta

#             flist_shifted = flist + v_offset_out
#             allv = np.vstack((allv, v_trans))
#             allf = np.vstack((allf, flist_shifted))
#             v_offset_out += len(v_trans)

#     else:
#         # ================= dist≠0分支：圆弧按短→长排序，中点沿X排布，移除对称旋转 =================
#         strip_arc_info = []
#         for s in strip_cache:
#             num = s["num"]
#             v = s["v"]
#             f = s["f"]
#             # 提取横截边中点polyline
#             mid_line = 0.5 * (v[:num] + v[num:2*num])
#             p0 = mid_line[0]       # 首横截边中点
#             pmid = mid_line[num//2]# 中间横截边中点(圆弧中点)
#             pend = mid_line[-1]    # 末横截边中点

#             # 三点拟合圆弧，计算圆弧长度
#             def fit_circle_3pts(p_a,p_b,p_c):
#                 A = 2*np.vstack([p_b-p_a, p_c-p_a])
#                 b = np.array([p_b@p_b - p_a@p_a, p_c@p_c - p_a@p_a])
#                 center, *_ = np.linalg.lstsq(A,b,rcond=None)
#                 r = np.linalg.norm(p_a-center)
#                 v1 = p_a-center
#                 v2 = p_c-center
#                 ang = np.arccos(np.clip(np.dot(v1,v2)/(r*r),-1,1))
#                 arc_len = r * ang
#                 return center, r, arc_len
#             c, r, arclen = fit_circle_3pts(p0, pmid, pend)
#             strip_arc_info.append({"strip":s, "arc_len":arclen, "pmid":pmid})

#         # 【圆弧由短到长排序】
#         strip_arc_info.sort(key=lambda d:d["arc_len"])

#         current_x = anchor
#         for item in strip_arc_info:
#             s = item["strip"]
#             num = s["num"]
#             vlist = s["v"]
#             flist = s["f"]
#             pmid_orig = item["pmid"]

#             # 只平移，删掉旋转对称代码，保留原有条带自身姿态不变
#             delta = np.array([current_x, 0.0, 0.0]) - pmid_orig
#             v_final = vlist + delta

#             # 网格拼接
#             flist_shifted = flist + v_offset_out
#             allv = np.vstack((allv, v_final))
#             allf = np.vstack((allf, flist_shifted))
#             v_offset_out += len(v_final)

#             # 更新下一个条带的x位置，间距=step*dist
#             current_x += step * dist

#     allsm = Mesh()
#     allsm.make_mesh(allv, allf)
#     return allsm


##above has orderred strips by their lengths; below has no new arrangements
# def unroll_multiple_strips(sm,num_list,dist=None,step=1.2,
#                            coo=2,anchor=0,efair=0.005,
#                            ee=0.001,itera=10,is_midaxis=False,
#                            w_straight=0,
#                            is_V_flip=False):
#     """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
#     Parameters
#     ----------
#     Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
#     anchor : one number of starting coordinates on x/y/z-axis
#     step : control the distance between different strips
#     coo : 0,1,2 ~ plane:y-z / x-z / x-y
#     """
#     Vall = sm.vertices
#     allv = np.empty((0,3), dtype=np.float64)   # 不再放dummy占位顶点
#     allf = np.empty((0,4), dtype=int)

#     istart_input = 0       # 原始输入Vall读取游标
#     v_offset_out = 0        # 输出allv顶点偏移，从0开始！！

#     for num in num_list:
#         Vi = Vall[istart_input:istart_input + 2*num]

#         if is_V_flip: ### no use now
#             "flip Vi from top to bottom"
#             Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))

#         _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)

#         # 无论是否成功展开，原始输入游标向前推进
#         istart_input += 2*num

#         if vlist is not None:
#             # ✅ 每个条带重新计算当前条带自身宽度
#             v1,v2 = vlist[0], vlist[num]
#             if dist is not None:
#                 dist_current = dist
#             else:
#                 dist_current = np.linalg.norm(v2-v1)
#             anchor += step * dist_current

#             # ✅ v_offset_out 是allv当前已有顶点总数，作为本批局部索引偏移
#             flist_shifted = flist + v_offset_out
#             allv = np.vstack((allv, vlist))
#             allf = np.vstack((allf, flist_shifted))

#             v_offset_out += len(vlist)

#     allsm = Mesh()
#     # allv,allf已经是干净数组，不需要切片[1:]
#     allsm.make_mesh(allv, allf)
#     return allsm


##above is debug by ai, below has a bug for different lengths of strips
# def unroll_multiple_strips_havebug(sm,num_list,dist=None,step=1.2,
#                            coo=2,anchor=0,efair=0.005,
#                            ee=0.001,itera=10,is_midaxis=False,
#                            w_straight=0,
#                            is_V_flip=False):
#     """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
#     Parameters
#     ----------
#     Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
#     anchor : one number of starting coordinates on x/y/z-axis
#     step : control the distance between different strips
#     coo : 0,1,2 ~ plane:y-z / x-z / x-y
#     """
#     #import time
#     #start_time = time.time()

#     Vall = sm.vertices
#     allv = np.array([0,0,0])
#     allf = np.array([0,0,0,0],dtype=int)
#     move = 0
#     istart = 0
#     for num in num_list:
#         Vi = Vall[istart:istart + 2*num]
#         if is_V_flip: ### no use now
#             "flip Vi from top to bottom"
#             Vi = np.vstack((Vi[:num][::-1],Vi[num:][::-1]))
#         _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
        
#         if vlist is not None: ##Hui new added for different lengths of strips
#             if dist is None:
#                 v1,v2 = vlist[0], vlist[num]
#                 dist = np.linalg.norm(v2-v1)
#             anchor += step*dist##NOTE: if +=step:const_controled_dist, if +=step*l, auto-dist
#             allv = np.vstack((allv,vlist))
#             allf = np.vstack((allf,flist+istart))
#             move = num*2
#             istart += move  
#     allsm = Mesh()
#     allsm.make_mesh(allv[1:],allf[1:])
    
#     #print("---Unroll %.2g seconds ---" % (time.time() - start_time))
#     return allsm

def unroll_1_strip(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight):
    """forked from meshFunction.py / unrollGeodesicStrip
        strip is constructed by two array of points [Varr1; Varr2]
        meshV: [col_i; col_i+1]
        coo: plane x=0, y=0, z=0, (coo=0,1,2)
        
        v4  --- v3
         |       |
         |       |
        v1 ---- v2
    """
    # import time
    # start_time = time.time()
    
    sm, v,f = opt_unfold_quad(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
    
    #print("---Unroll %.2g seconds ---" % (time.time() - start_time))
    
    return sm,v,f

def distortion_all_strips(mesh3d,mesh2d,edge=False,length=False,angle=False,area=False,dig_ctrl=False):
    v3d, v2d = mesh3d.vertices, mesh2d.vertices,
    flist = np.array(mesh3d.faces_list())
    v1,v2,v3,v4 = v3d[flist[:,0]],v3d[flist[:,1]],v3d[flist[:,2]],v3d[flist[:,3]]
    w1,w2,w3,w4 = v2d[flist[:,0]],v2d[flist[:,1]],v2d[flist[:,2]],v2d[flist[:,3]]
    el = mesh3d.mean_edge_length()    
    d11 = np.linalg.norm(v3-v1,axis=1)
    d12 = np.linalg.norm(v4-v2,axis=1)
    d21 = np.linalg.norm(w3-w1,axis=1)
    d22 = np.linalg.norm(w4-w2,axis=1)  
        
    if edge:
        i1,i2 = mesh3d.edge_vertices()
        dv = np.linalg.norm(v3d[i1]-v3d[i2],axis=1)
        dw = np.linalg.norm(v2d[i1]-v2d[i2],axis=1)
        err = (np.abs(dv-dw)) / el
        
    elif length:
        "diagonal length of control face"    
        err = (np.abs(d11-d21)+np.abs(d12-d22)) / el
        print('\n Isometry (length) error = (np.abs(d11-d21)+np.abs(d12-d22)) / el : \n')

    elif angle:
        ag1 = np.einsum('ij,ij->i',v3-v1,v4-v2)
        ag2 = np.einsum('ij,ij->i',w3-w1,w4-w2)
        cos1 = ag1/d11/d12
        cos2 = ag2/d21/d22
        try:
            err = np.abs((np.arccos(cos1)-np.arccos(cos2))*180/np.pi)
        except:
            err = np.zeros(len(cos1))
            for i in range(len(cos1)):
                if cos1[i]>1 or cos1[i]<-1 or cos2[i]>1 or cos2[i]<-1:
                    print(cos1[i],cos2[i])
                    continue
                else:
                    err[i] = np.abs((np.arccos(cos1[i])-np.arccos(cos2[i]))*180/np.pi)
        print('\n Isometry (angle) error=np.abs((np.arccos(cos1)-np.arccos(cos2))*180/np.pi) : \n')
        
    elif area:
        a1 = np.linalg.norm(np.cross(v2-v1,v4-v1),axis=1)
        a2 = np.linalg.norm(np.cross(w2-w1,w4-w1),axis=1)
        err = np.abs(a1-a2) / (el**2) / 2
        print('\n Isometry (area) error : \n')

    elif dig_ctrl:
        "diagonal length of checkerboard quad"
        d11 = np.linalg.norm((v2+v1)/2-(v3+v4)/2,axis=1)
        d12 = np.linalg.norm((v3+v2)/2-(v4+v1)/2,axis=1)
        d21 = np.linalg.norm((w2+w1)/2-(w3+w4)/2,axis=1)      
        d22 = np.linalg.norm((w3+w2)/2-(w4+w1)/2,axis=1)
        err = (np.abs(d11-d21)+np.abs(d12-d22))/2 / el

    print('   max = ','%.3g' % np.max(err),'\n')    
    return err
    
# -------------------------------------------------------------------------
#          Unroll optmization : ARAP - ISOMETRY CONSTRAINTS:
# -------------------------------------------------------------------------
def opt_unfold_quad(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight):
    #start_time = time.time()
    
    num = int(len(V)/2)
    #print(num)
    
    if num > 2: ##Hui new added for different lengths of strips
        X,var,lh,ll,lr,l13,l24,fixp = initial_unfold(V,coo,anchor,is_midaxis,w_straight)
        n = 0
        K = matrix_fair(num,var,efair)
        I = sparse.eye(var,format='coo')*ee**2
    
        opt_num, opt = 100, 100
        "note should have opt-check, otherwise only run once for asymtotpicstrips"
        while n < itera and (opt_num>1e-6 or opt>1e-6) and opt_num<1e+6:
        #while n < itera and opt_num>0.00001 and opt_num<1e+6:
            H, r, opt = con_isometry(X,num,lh,ll,lr,l13,l24,coo,anchor,fixp,is_midaxis,w_straight)
            X = sparse.linalg.spsolve(H.T*H+K.T*K+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
            n += 1
            opt_num = np.sum(np.square((H*X)-r))
            #print('Optimize the unfolding mesh:',n, '%.2g' %opt, '%.2g' %opt_num)#'%.2g' %opt_num,
        
        #print(n, '%.2g' %opt_num),#'%.2g s' %(time.time() - start_time)) ##hui comment/uncomment
        
        Vl = X[:3*num].reshape(-1,3,order='F')
        Vr = X[3*num:6*num].reshape(-1,3,order='F')
        V = np.vstack((Vl,Vr))
        arr = np.arange(num)
        v1,v2,v3,v4 = arr[:-1],arr[:-1]+num,arr[1:]+num,arr[1:]
        flist = np.c_[v1,v2,v3,v4]
        sm = Mesh()
        sm.make_mesh(V, flist)    
        return sm,V,flist
    else:
        return None, None, None

def initial_unfold(V,coo,anchor,is_midaxis,w_straight):
    """ X = [Vl ;  Vr]
        v4  --- v3
         |       |
         |       |
        v1 ---- v2    
    """
    num = int(len(V)/2)
    vl,vr = V[:num],V[num:]
    v1,v4 = vl[:-1],vl[1:]
    v2,v3 = vr[:-1],vr[1:]
    lh = np.linalg.norm(vl-vr,axis=1) # horizontal
    ll = np.linalg.norm(v1-v4,axis=1) # left
    lr = np.linalg.norm(v2-v3,axis=1) # right
    l13 = np.linalg.norm(v1-v3,axis=1) 
    l24 = np.linalg.norm(v2-v4,axis=1)     
    
    d1,d2 = [0],[0]
    k1,k2 = 0,0
    for i in range(num-1):
        k1 += ll[i]
        k2 += lr[i]
        d1.append(k1)
        d2.append(k2)
    Vl,Vr = np.array(d1), np.array(d2)

    dist_axis = lh[0]
    zero = np.zeros(num)
    mid = (Vl[-1]+Vr[-1])/4
    if is_midaxis:
        if coo==0:
            "put in y-axis: y-z plane"
            fixp = np.array([0,anchor,0])
            origin = np.array([0,anchor-dist_axis/2,-mid])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero,zero+dist_axis,Vr]
        elif coo==1:
            "put in x-axis: x-z plane"
            fixp = np.array([anchor, 0, 0])
            origin = np.array([anchor-dist_axis/2, 0, -mid])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero+dist_axis,zero,Vr]
        elif coo==2:
            "put in x-axis: x-y plane"
            fixp = np.array([anchor, 0, 0])
            origin = np.array([anchor-dist_axis/2, -mid, 0])
            Pl = origin + np.c_[zero,Vl,zero]
            Pr = origin + np.c_[zero+dist_axis,Vr,zero]
    else:    
        if coo==0:
            "put in y-axis: y-z plane"
            origin = np.array([0,anchor,0])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero,zero+dist_axis,Vr]
        elif coo==1:
            "put in x-axis: x-z plane"
            origin = np.array([anchor, 0, 0])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero+dist_axis,zero,Vr]
        elif coo==2:
            "put in x-axis: x-y plane"
            origin = np.array([anchor, -mid, 0]) ##NOTE: ORIGIN=[an,0,0] or [an,-mid,0]
            Pl = origin + np.c_[zero,Vl,zero]
            Pr = origin + np.c_[zero+dist_axis,Vr,zero]
        fixp = np.r_[Pl[0],Pr[0]]
        
    X = np.r_[Pl.flatten('F'),Pr.flatten('F')]    
    if w_straight:
        "right-const.coordinate"
        if coo==0:
            lconst,rconst = np.mean(Pl[:,1]),np.mean(Pr[:,1])
        elif coo==1:
            lconst,rconst = np.mean(Pl[:,0]),np.mean(Pr[:,0])
        elif coo==2:
            lconst,rconst = np.mean(Pl[:,0]),np.mean(Pr[:,0])
        X = np.r_[X,lconst,rconst]
    var = len(X)
    return X,var,lh,ll,lr,l13,l24,fixp

def con_isometry(X,num,lh,ll,lr,l13,l24,coo,anchor,fixp,is_midaxis,w_straight):
    arr = np.arange(num)
    c_vl = np.r_[arr, num+arr, 2*num+arr]
    c_vr = c_vl + 3*num
    
    c_v1 = np.r_[arr[:-1], num+arr[:-1], 2*num+arr[:-1]]
    c_v4 = np.r_[arr[1:], num+arr[1:], 2*num+arr[1:]]
    c_v2, c_v3 = c_v1 + 3*num, c_v4 + 3*num

    def _con_in_plane():
        "coo=0,1,2 ~ yz, xz, xy -- plane"
        row = np.arange(2*num)
        col = coo*num+np.r_[arr,3*num+arr]
        data = np.ones(2*num)
        H = sparse.coo_matrix((data,(row,col)), shape=(2*num, len(X)))
        r = np.zeros(2*num)
        return H, r  
    
    def _con_edge_length():
        def _edge(c1,c2,l1):
            "(P1-P2)^2 = l1**2"
            num = len(l1)
            arr = np.arange(num)
            col = np.r_[c1,c2]
            row = np.tile(arr,6)
            data = 2*np.r_[X[c1]-X[c2],X[c2]-X[c1]]
            r = np.linalg.norm((X[c1]-X[c2]).reshape(-1,3, order='F'),axis=1)**2+l1**2
            H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
            return H,r   
        H1,r1 = _edge(c_vl,c_vr,lh)
        H2,r2 = _edge(c_v1,c_v4,ll)
        H3,r3 = _edge(c_v2,c_v3,lr)
        H4,r4 = _edge(c_v1,c_v3,l13)
        H5,r5 = _edge(c_v2,c_v4,l24)
        H = sparse.vstack((H1*10,H2*10,H3*10,H4*10,H5*10)) ##NOTE: NEED TO SET OF WEIGHTS
        r = np.r_[r1*10,r2*10,r3*10,r4*10,r5*10]
        return H ,r
    
    def _con_fix():
        "fix the axis-alignment two points"
        row = np.arange(6)
        col = np.r_[c_vl[np.array([0,num,2*num])], c_vr[np.array([0,num,2*num])]]
        data = np.ones(6)
        r = fixp
        H = sparse.coo_matrix((data,(row,col)), shape=(6, len(X)))
        return H * 10, r * 10
    
    def _con_fix2():
        """fix the midaxis-alignment of central point of each strip
        mid = (vl0+vl1+vr0+vr1)/4;
        p=(vl0+vr0)/2 
            if coo=0,yz-plane,y-axis, p[1] == mid[1]
            if coo=1,xz-plane,x-axis, p[0] == mid[0]
            if coo=2,xy-plane,x-axis, p[0] == mid[0]
        """
        def _fix(c_a,c_b,c_c,c_d):
            "va+vb+vc+vd = 4*fixp"
            row = np.tile(np.arange(3),4)
            col = np.r_[c_a,c_b,c_c,c_d]
            data = np.ones(12)
            r = fixp * 4
            H = sparse.coo_matrix((data,(row,col)), shape=(3, len(X)))
            return H * 10, r * 10
        
        def _glide(c_i,c_j,x):
            "([vl0+vr0])/2=x <==> [vl0+vr0][i] = 2x"
            row = np.array([0,0])
            col = np.array([c_i,c_j])
            data = np.array([1,1])
            r = np.array([2*x])
            H = sparse.coo_matrix((data,(row,col)), shape=(1, len(X)))
            return H,r
        if coo==0:
            i = 1
        elif coo==1 or coo==2:
            i = 0
        arr = np.array([0,num,2*num])
        c_i, c_j, x = c_vl[arr][i], c_vr[arr][i], fixp[i]
        H1,r1 = _glide(c_i,c_j,x)
        c_a = c_vl[np.array([0,num,2*num])]
        c_b = c_vr[np.array([0,num,2*num])]   
        c_c = c_vl[np.array([num-1,2*num-1,3*num-1])]
        c_d = c_vr[np.array([num-1,2*num-1,3*num-1])]
        H2,r2 = _fix(c_a,c_b,c_c,c_d)
        H = sparse.vstack((H1,H2))
        r = np.r_[r1,r2]
        return H,r
    
    def _con_strong_straight():
        "1.vertical-coordiante of left+right is constant"
        "2. global lenght of left/right is kept"
        def _global_edgelength(c_vl,ll,c_const):
            " "
           # Ll = np.sum(ll)
            # if is_midaxis:
            #     null =  np.zeros([0])
            #     H = sparse.coo_matrix((null,(null,null)), shape=(0,var))
            #     r = np.array([])
            # else:
            if coo==0:
                "put in y-axis: y-z plane"
                pass
            elif coo==1:
                "put in x-axis: x-z plane"
                pass
            elif coo==2:
                "put in x-axis: x-y plane, [x,y,0]"
                "difference(endcoordiante)=global_length" ###NO USE NOW!
                # row = np.zeros(2)
                # col = np.array([c_vl[2*num],c_vl[num]]) ##y[-1]-y[0]
                # data = np.array([1,-1],dtype=int)
                # Hg = sparse.coo_matrix((data,(row,col)), shape=(1, var))
                # rg = np.array([Ll]) 
                "vertical_coordinate == anchor"
                row = np.tile(arr,2)
                col = np.r_[c_vl[:num],np.ones(num)*c_const] #x
                data = np.r_[np.ones(num),-np.ones(num)]
                rc = np.zeros(num)
                Hc = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
                #H = sparse.vstack((Hg,Hc))
                #r = np.r_[rg,rc]
            return Hc,rc
        Hs1,rs1 = _global_edgelength(c_vl,ll,len(X)-2)
        Hs2,rs2 = _global_edgelength(c_vr,lr,len(X)-1)
        H = sparse.vstack((Hs1,Hs2))
        r = np.r_[rs1,rs2]
        return H,r

    H1,r1 = _con_in_plane()
    H2,r2 = _con_edge_length()
    "NOTE:BELOW COMMENT FOR EQUAL STRIPS, NEED TO CHECK FOR OTHERS"
    # if is_midaxis:
    #     H3,r3 = _con_fix2()
    # else:
    #     H3,r3 = _con_fix()
    # H = sparse.vstack((H1,H2,H3))
    # r = np.r_[r1,r2,r3]
    H = sparse.vstack((H1,H2))
    r = np.r_[r1,r2]
    
    if w_straight:
        Hs,rs = _con_strong_straight()
        H = sparse.vstack((H,Hs*w_straight))
        r = np.r_[r,rs*w_straight]
    
    opt = np.sum(np.square((H*X)-r))
    return H,r,opt

def matrix_fair(num,var,efair=0.005):
    "for vl and vr polyline"
    arr = np.arange(num)
    c_vl = np.r_[arr, num+arr, 2*num+arr]
    c_vr = c_vl + 3*num
    #num = int(len(c_vl)/3)
    c_l1 = np.r_[c_vl[:num-2],c_vl[num:2*num-2],c_vl[2*num:3*num-2]]
    c_r1 = np.r_[c_vr[:num-2],c_vr[num:2*num-2],c_vr[2*num:3*num-2]]
    c_l2 = np.r_[c_vl[1:num-1],c_vl[num+1:2*num-1],c_vl[2*num+1:3*num-1]]
    c_r2 = np.r_[c_vr[1:num-1],c_vr[num+1:2*num-1],c_vr[2*num+1:3*num-1]]
    c_l3 = np.r_[c_vl[2:num],c_vl[num+2:2*num],c_vl[2*num+2:3*num]]
    c_r3 = np.r_[c_vr[2:num],c_vr[num+2:2*num],c_vr[2*num+2:3*num]]
    c_1 = np.r_[c_l1,c_r1]
    c_2 = np.r_[c_l2,c_r2]
    c_3 = np.r_[c_l3,c_r3]
    num = len(c_1)
    row = np.tile(np.arange(num),3)
    col = np.r_[c_1,c_2,c_3]
    data = np.r_[np.ones(num),-2*np.ones(num),np.ones(num)]
    K = sparse.coo_matrix((data,(row,col)), shape=(num, var))
    return efair * K



# Below on use, only refer:
# -------------------------------------------------------------------------
#          Unroll functions from GO-project: meshFunctions.py
# -------------------------------------------------------------------------

def unrollGeodesicStrip(V, coo=2, anchor=0, sign=-1):
    """forked from meshFunction.py / 
        strip is constructed by two array of points [Varr1; Varr2]
        meshV: [col_i; col_i+1]
        coo: plane x=0, y=0, z=0, (coo=0,1,2)
        
        v1 --- v22
         |       |
         |       |
        v0 ---- v21
    """    
    num = int(len(V)//2)
    L10 = np.linalg.norm(V[1:num,:]-V[:num-1,:],axis=1) #|v1-v0|
    L121 = np.linalg.norm(V[1:num,:]-V[num:2*num-1,:],axis=1) #|v1-v21|
    L221 = np.linalg.norm(V[num+1:2*num,:]-V[1:num,:],axis=1) #|v22-v1|
    L2221 = np.linalg.norm(V[num+1:2*num,:]-V[num:2*num-1,:],axis=1) #|v22-v21|

    # put w0, w21 in x-axis
    v0, v21 = V[0], V[num]
    l210 = np.linalg.norm(v21-v0)

    if coo==0:# put in y-axis: y-z plane
        w0 = np.array([0,anchor,0])
        w21 = w0 + np.array([0,l210,0])
        a,b = 1,2
    elif coo==1:# put in x-axis: x-z plane
        w0 = np.array([anchor, 0, 0])
        w21 = w0 + np.array([l210,0,0])
        a,b = 0,2
    elif coo==2:# put in x-axis: x-y plane
        w0 = np.array([anchor, 0, 0])
        w21 = w0 + np.array([l210,0,0])
        a,b=0,1
    Vl, Vr = w0, w21
    def stripsV(i, w0, w21):
        "get the whole strip vertices"
        # solve order: w0, w21, w1, w22; ....
        # w1=[x1,y1,0] w22=[x22,y22,0]
        l121 = L121[i-1]
        l10 = L10[i-1]
        l221 = L221[i-1]
        l2221 = L2221[i-1]
        # kxl, kxr, ky are very important to impact result
        kxl, kxr = w0[a], w21[a]
        kyl = w0[b] + l10 # may be not plus
        kyr = w21[b] + l2221
        def vv1(x): # x are the variables of the equation
            return np.array([(x[0]-w21[a])**2+(x[1]-w21[b])**2-l121**2,
                             (x[0]-w0[a])**2+(x[1]-w0[b])**2-l10**2])
        w1_solve = fsolve(vv1,[kxl,kyl])
        w1 = np.insert(w1_solve,coo,0)

        def vv22(x):
            return np.array([(x[0]-w21[a])**2+(x[1]-w21[b])**2-l2221**2,
                             (x[0]-w1[a])**2+(x[1]-w1[b])**2-l221**2])
        w22_solve = fsolve(vv22,[kxr,kyr])
        w22 = np.insert(w22_solve,coo,0)
        return w1, w22

    for i in np.arange(num-1)+1: #[1,2,..20]
        w0, w21 = stripsV(i,w0,w21)
        Vl = np.vstack((Vl, w0))
        Vr = np.vstack((Vr, w21))

    vlist = sign * np.vstack((Vl, Vr))
    arr = np.arange(num)
    v1,v2,v3,v4 = arr[:-1],arr[:-1]+num,arr[1:]+num,arr[1:]
    flist = np.c_[v1,v2,v3,v4]
    sm = Mesh()
    sm.make_mesh(vlist, flist)
    return sm,vlist,flist


def unrollStripMesh(self, col_index, coo):
    "for a strip"
    "meshV: [col_i; col_i+1]"
    sMeshFL = self.faceStripReindex().tolist()
    vSR = self.vertexStripIndex(col_index)
    V = self.vertices[vSR]
    sMeshVL = self.unrollGeodesicStrip(V, coo).tolist()
    unrollMesh = Mesh()
    unrollMesh.make_mesh(sMeshVL, sMeshFL)
    return unrollMesh

def unfoldAllGeodesicStrips(self, anchor=0, step=1, coo=2):
    "above unroll a strip, this unrolls all strips"
    col_indices = np.arange(self.colnum+1)
    vertexArray = np.array([0,0,0],dtype=int)
    faceArray=np.array([0,0,0,0],dtype=int)
    fIndex = self.faceStripReindex()
    for i in col_indices:#[0,1,2,...19]
        vSR = self.vertexStripIndex(i)
        V = self.vertices[vSR]
        vlist = self.unrollGeodesicStrip(V, coo, anchor)# len(vIndex)=42*20,rownum=colnum=20

        v0, v21 = V[0], V[len(V)//2]
        l210 = np.linalg.norm(v21-v0)
        anchor += step * l210
        vertexArray = np.vstack((vertexArray, vlist))
        faceArray = np.vstack((faceArray, fIndex+i*(1+np.max(fIndex))))

    sVlist = vertexArray[1:,:].tolist()
    sFlist = faceArray[1:,:].tolist()
    unrollMesh = Mesh()
    unrollMesh.make_mesh(sVlist, sFlist)
    return unrollMesh

def unfoldWholeMeshAIAP(self,Vv,eFair,ee,coo):
    "use file from unfoldingMesh"
    Y = self.opt_unfoldMesh2Plane(Vv,eFair,ee,coo)
    vertices = (Y[:3*self.V].reshape(-1,3)).tolist()
    faces = self.faces_list()
    wmesh = Mesh()
    wmesh.make_mesh(vertices,faces)
    return wmesh

def unfoldParallelogramMeshAIAP(self,ee):
    "use file unfoldingParallelogram (like unfoldWholeMeshAIAP)"
    V = self.vertices
    vM = self.vertexIndexMatrix()
    overlap=False
    Y = self.opt_unfoldParallelogramMesh2Plane(V,vM,overlap,ee,coo=2)
    self.unfoldY = Y
    vertices = (Y[:3*self.V].reshape(-1,3)).tolist()
    faces = self.faces_list()
    pmesh = Mesh()
    pmesh.make_mesh(vertices,faces)
    return pmesh

def unfoldParallelogram_Mesh(self):
    "use file unfoldingParallelogram (like unfoldWholeMeshAIAP)"
    if self.unfoldY is not None:
        Y = self.unfoldY
        vertices = (Y[3*self.V:].reshape(-1,3)).tolist()
        num = len(vertices)//4
        fM = (np.arange(4*num).reshape(-1,num)).T
        flist = fM.tolist()
        pmesh = Mesh()
        pmesh.make_mesh(vertices,flist)
        return pmesh
    
def unfoldOverlapParallelogramMesh(self,ee):# second optimize\showing
    "project all parallelogram to z=0, minimize the overlap points"
    V = self.vertices
    vM = self.vertexIndexMatrix()
    overlap = True
    Y = self.opt_unfoldParallelogramMesh2Plane(V,vM,overlap,ee,coo=2)
    n = 3*self.V
    P1 = Y[n:n+3*self.fnum].reshape(-1,3)
    P2 = Y[3*self.fnum+n:n+6*self.fnum].reshape(-1,3)
    P3 = Y[6*self.fnum+n:n+9*self.fnum].reshape(-1,3)
    P4 = Y[9*self.fnum+n:n+12*self.fnum].reshape(-1,3)
    return P1,P2,P3,P4        
        