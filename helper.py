import tools
from scipy.optimize import linear_sum_assignment
import numpy as np
import mmcv

import sys
# sys.path.append('/root/py-mcftracker/player-feature-extractor')
# sys.path.append('/root/bepro-python')
sys.path.append('/home/bepro/bepro-python')

import torch
from pfe.torchreid.utils import FeatureExtractor
from pfe.scripts.extract_fetures import network_feed_from_list

from bepy.transform import Transform
from bepy.models import MatchVideo

import cv2
from scipy.spatial import distance

import debug
import copy

import matplotlib.pyplot as plt
from scipy.misc import face

import math

from node import GraphNode
from scipy import interpolate

def isGoalArea(transform, xwc, size=None, frame=None):

    # gp = [[0, 54.16, 0], [16.5, 54.16, 0], [0, 13.84, 0], [16.5, 13.84, 0],
    #         [105, 54.16, 0], [88.5, 54.16, 0], [105, 13.84, 0], [88.5, 13.84, 0]]

    # for p in gp:
    #     x,y = transform.ground_to_video(p[0]/105., p[1]/68., 0)
    #     x = x*size[1]
    #     y = y*size[0]
    #     cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), 5)

    # cv2.imwrite('_check_goal_area.jpg', frame)

    x, y = xwc[0], xwc[1]

    x = x * transform.parameter.get("ground_width")
    y = y * transform.parameter.get("ground_height")

    if (x >= 0 and x<= 16.5 and y>=13.84 and y<=54.16) or (x>=88.5 and x<=105 and y>=13.84 and y<=54.16):
        return True

    return False

def box2midpoint_normalised(box, iw, ih):
    w = box[2]-box[0]
    x, y = box[0] + w/2, box[3]
    return (x/iw, y/ih)

def is_box_occluded(tlbr, boxes, t_iou=1.0):
    # calculate iou with all boxes in current frame
    for box in boxes:
        _, x1, y1, x2, y2, s = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5])
        cb = [x1,y1,x2,y2]
        
        if cb == tlbr:
            continue

        if tools.calc_overlap(tlbr, cb) > t_iou:
            return True

    return False

def is_patch_complex_scene(index, wc, transform, tdist=5.0, tcrowd=5):
    crowd = 0
    cb = wc[index]
    cx, cy = cb[0], cb[1]

    for i, b in enumerate(wc):
        if i == index:
            continue

        rx, ry = b[0], b[1]
        dist = calc_eucl_dist([cx*transform.parameter.get("ground_width"),cy*transform.parameter.get("ground_height")], 
                            [rx*transform.parameter.get("ground_width"),ry*transform.parameter.get("ground_height")])

        if dist <= tdist:
            crowd = crowd+1

    if crowd >= tcrowd:
        return True

    return False

def find_prev_imgbox(transform, box_cur, detections, images, name, frame, bi):
    prev_index = str(int(name)-1)
    # max_iou_index = debug.return_closest_box_index(box_cur, detections[prev_index], frame, bi)
    min_dist_index = debug.return_closest_box_index_extrinsic(transform, box_cur, detections[prev_index], frame, bi)

    if min_dist_index == -1:
        return False, None

    patch = copy.deepcopy(images[prev_index][min_dist_index])
    return True, patch

def convert2world(rows, size, transform):
    wc = []
    for n,r in enumerate(rows):
        _, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
        cb = [x1,y1,x2,y2,s]
        p = box2midpoint_normalised(cb, size[1], size[0])
        cx, cy = transform.video_to_ground(p[0], p[1])
        # print (cx,cy)
        wc.append((cx,cy))
    return wc

def convert2world_post(rows, size, transform):
    wc = []
    for n,r in enumerate(rows):
        _, x1, y1, w, h, _ = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[9])
        cb = [x1,y1,x1+w,y1+h]
        p = box2midpoint_normalised(cb, size[1], size[0])
        cx, cy = transform.video_to_ground(p[0], p[1])
        wc.append((cx,cy))
    return wc

def read_input_data(path2det, path2video, slice_start, slice_end, det_in, frame_indices, match_video_id,
                        # ckpt_path='/root/py-mcftracker/player-feature-extractor/checkpoints/market_combined_120e.pth'):
                        ckpt_path='/home/bepro/py-mcftracker/pfe/checkpoints/market_combined_120e.pth'):
    
    input_data = {}
    video = mmcv.VideoReader(path2video)

    # testing torche reid import
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=ckpt_path,
        device='cuda'
    )

    match_video = MatchVideo.load(match_video_id)

    transform = Transform(
        match_video.video.camera_recording["parameter"],
        match_video.video.camera_recording["extrinsic_json"],
        match_video.video.camera_recording["stitching_json"],
    )

    size = np.zeros(2)
    fnum = 0

    print ('-> video frame interval [%s-%s]' % (slice_start, slice_end-1))

    parity = False

    if slice_start %2 != 0 and (slice_end-1) % 2 != 0: # odd - odd
        parity = True
    elif slice_start %2 == 0 and (slice_end-1) % 2 == 0: # even - even
        parity = True
        
    for index in range(slice_start, slice_end):
        frame = video[index]

        if index == slice_start:
            size = frame.shape
            cv2.imwrite("./frame.jpg", frame)

        # skip frame
        # note: if index of last frame in chunk is even, throw away odd indexes and vice versa
        # frame of last index is needed for inter-chunk connection
        if parity:
            if (index-slice_start) % 2 != 0:
                continue
        else:
            # instead of skipping 0 index skip 1 index
            if (index-slice_start) != 0 and (index-slice_start) % 2 == 0:
                continue
            if (index-slice_start) == 1:
                continue

        fnum = fnum+1

        if fnum % 500 == 0:
            print ('-> reading frame %d / %d' % (fnum, slice_end-slice_start))

        mask = frame_indices == (index - slice_start + 1)
        # mask = frame_indices == fnum
        rows = det_in[mask]

        image_name = "%d" % (fnum)

        bbimgs = []
        node_lst = []
 
        _wc = convert2world(rows, size, transform)

        for n,r in enumerate(rows):
            _, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])

            curbox = [x1,y1,x2,y2]
            imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            node = GraphNode(_wc[n], curbox, s, 0, copy.deepcopy(imgbox))

            # if is_box_occluded(node._bb, rows, t_iou=0.10):
            #     if is_patch_complex_scene(n, _wc, transform, tdist=5.0):
            #         if isGoalArea(transform, node._3dc, size, frame):
            #             if is_patch_complex_scene(n, _wc, transform, tdist=3.0):
            #                 node._status = 4
            #             else:
            #                 node._status = 2
            #         else:
            #             if is_patch_complex_scene(n, _wc, transform, tdist=2.0):
            #                 node._status = 3
            #             else:
            #                 node._status = 2
            #     else:
            #             node._status = 1

            bbimgs.append(imgbox)
            node_lst.append(node)
            
        if len(bbimgs) == 0:
            print ('no images')
            continue

        feats = network_feed_from_list(bbimgs, extractor)

        for i,node in enumerate(node_lst):
            node._feat = feats[i]
        input_data[image_name] = node_lst

    print ('-> %d images have been read & processed' % (len(input_data)))

    return input_data, transform, size, parity

def write_output_data(track_hypot, path2det, data, iend, frame_offset, iid, parity):
    # write to file
    log_filename = './hypothesis.txt'
    log_file = open(log_filename, 'w')

    f = 1
    all_lines = []

    for n in range(1, iend):
        lines = []
        for id, track in enumerate(track_hypot):
            for i, t in enumerate(track):
                if i % 2 == 0:
                    
                    if int(t[0]) == n:
                        bi = int(t[1])
                        b = data[t[0]][bi]._bb
                        s = data[t[0]][bi]._status
                        # must be in top-left-width-height
                        lines.append([f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], s])
                        # log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], 1))

        all_lines.append(lines)
        f = f+2

    print (len(all_lines))

    if not parity:
        for i in range(len(all_lines)-1):
            fl = sorted(all_lines[i], key=lambda x: x[1])
            sl = sorted(all_lines[i+1], key=lambda x: x[1])

            al = []

            for l in fl:
                for m in sl:
                    if m[1] == l[1]:
                        al.append([(l[k] + m[k])/2 if k!=len(l)-1 else l[k] for k in range(len(l))])

            for l in fl:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))

            for l in al:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))
        
        # last index
        li = len(all_lines)-1
        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))

        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0]+1, l[1], l[2], l[3], l[4], l[5], l[6]))

    else:
        for i in range(len(all_lines)-1):
            fl = sorted(all_lines[i], key=lambda x: x[1])
            sl = sorted(all_lines[i+1], key=lambda x: x[1])

            al = []

            for l in fl:
                for m in sl:
                    if m[1] == l[1]:
                        al.append([(l[k] + m[k])/2 if k!=len(l)-1 else l[k] for k in range(len(l))])

            for l in fl:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))

            for l in al:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))

        # write last line
        li = len(all_lines)-1
        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))

def extract_patch_block(patch):
    h, w = patch.shape[0], patch.shape[1]
    half_p = patch[:int(h/2), :, ]
    s = int(0.3 * w)
    roi = half_p[:, s:w-s, :]
    return roi

def recursive_get_track(elem, dct, lst):
    if elem in dct:
        lst.append(elem)
        k = dct[elem].items()
        recursive_get_track(k[0][0], dct, lst)
    return

def loop_get_track(elem, dct):
    lst = []
    # while dict has key
    while elem in dct:
        lst.append(elem)
        elem = list(dct[elem].keys())[0]
    return lst

def calc_eucl_dist(det1, det2):
    dist = 0
    pt1 = det1[:2]
    pt2 = det2[:2]
    pt1_np = np.asarray(pt1, dtype=np.float)
    pt2_np = np.asarray(pt2, dtype=np.float)
    dist = np.linalg.norm(pt1_np-pt2_np)
    return dist

def return_max_dist(x):
    return math.log(x) + 2.8

def compute_cost(u, v, cur_box, ref_box, transform, size, frame_gap, alpha=0.6, inf=1e6):

    if frame_gap < 1:
        return inf
        
    cos_dist = distance.cosine(u, v)

    # test: project points (0,0,0), (1.0,0,0), (0,1.0,0), (1.0,1.0,0) to image
    # frame = cv2.imread("frame.jpg")
    # ground_points = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,0.]]
    # for point in ground_points:        
    #     x, y = transform.ground_to_video(point[0], point[1])
    #     x = x * frame.shape[1]
    #     y = y * frame.shape[0]
    #     cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), 5)
    #cv2.imwrite("frame.jpg", frame)

    p1 = box2midpoint_normalised(cur_box, size[1], size[0])
    p2 = box2midpoint_normalised(ref_box, size[1], size[0])

    cx, cy = transform.video_to_ground(p1[0], p1[1])
    rx, ry = transform.video_to_ground(p2[0], p2[1])

    # print (transform.parameter.get("ground_width"), transform.parameter.get("ground_height"))

    dist = calc_eucl_dist([cx*transform.parameter.get("ground_width"),cy*transform.parameter.get("ground_height")], 
                        [rx*transform.parameter.get("ground_width"),ry*transform.parameter.get("ground_height")])

    # maxdistance = return_max_dist(frame_gap)
    maxdistance = 4.

    if dist > maxdistance: 
        return inf

    dist_norm = dist / maxdistance
    cost = alpha*dist_norm + (1-alpha)*cos_dist

    return cost

def cost_matrix(hypothesis, hypothesis_t, hypothesis_s, data, transform, size, inf=1e6, max_gap=100):
    cost_mtx = np.zeros((len(hypothesis_t), len(hypothesis_s)))

    for i, index_i in enumerate(hypothesis_t):
        for j, index_j in enumerate(hypothesis_s):
            last_idx = hypothesis[index_i][-1] # tuple ('frame_num', detection_index, 'u')
            first_idx = hypothesis[index_j][0]

            gap = int(first_idx[0]) - int(last_idx[0])
            
            feat_tail = data[last_idx[0]][last_idx[1]]._feat
            feat_head = data[first_idx[0]][first_idx[1]]._feat

            det_tail = data[last_idx[0]][last_idx[1]]._bb
            det_head = data[first_idx[0]][first_idx[1]]._bb

            cost_mtx[i][j] = compute_cost(feat_tail, feat_head, det_tail, det_head, transform, size, gap)
            
            # if data[last_idx[0]][last_idx[1]]._status == 3 or data[last_idx[0]][last_idx[1]]._status == 4:
            #     cost_mtx[i][j] = inf

            # check if gap isn't too large
            if gap >= max_gap:
                cost_mtx[i][j] = inf

    return cost_mtx

def temporal_hungarian_matching(hypothesis, hypothesis_t, hypothesis_s, data, transform, size):
    cost = cost_matrix(hypothesis, hypothesis_t, hypothesis_s, data, transform, size)
    
    # assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    print ('temporal hungarian assignment')

    matches = []
    for r,c in zip(row_ind, col_ind):
        if cost[r][c] != 1e6:
            print ('id %d -> id %d - frames: [%d-%d] cost: %f' % (hypothesis_t[r]+1, hypothesis_s[c]+1, 
                        2*int(hypothesis[hypothesis_t[r]][-1][0]), 2*int(hypothesis[hypothesis_s[c]][0][0]), cost[r][c]))
            matches.append((hypothesis_t[r], hypothesis_s[c]))

    # sort matches in descending order of hypothesis_s
    matches.sort(key=lambda tup: tup[1], reverse=True)
    print ('matches after sorting ==>')
    print (matches)
    
    # for s,e in matches:
    #     ns = hypothesis[s][-1] # tuple ('frame_num', detection_index, 'u')
    #     ne = hypothesis[e][0]

    #     fs = int(ns[0])
    #     fe = int(ne[0])
  
    #     bs = data[ns[0]][ns[1]]._bb
    #     be = data[ne[0]][ne[1]]._bb

    #     # print ('%d -> %d > %s -> %s' % (fs,fe,bs,be))

    #     fpx1 = [bs[0], be[0]]
    #     fpy1 = [bs[1], be[1]]
    #     fpx2 = [bs[2], be[2]]
    #     fpy2 = [bs[3], be[3]]

    #     xp = [fs, fe]

    #     for x in range(fs+1, fe):
    #         pix1 = np.interp(x, xp, fpx1)
    #         piy1 = np.interp(x, xp, fpy1)
    #         pix2 = np.interp(x, xp, fpx2)
    #         piy2 = np.interp(x, xp, fpy2)

    #         # print ('%d -> %s' % (x, [pix1,piy1,pix2,piy2]))
    #         pi = [pix1,piy1,pix2,piy2]

    #         gn = GraphNode([0.,0.], pi, 0., 0)
    #         fn = str(x)

    #         index = len(data[fn])
    #         data[fn].append(gn)

    #         node_u = (fn, index, "u")
    #         node_v = (fn, index, "v")

    #         hypothesis[s].append(node_u)
    #         hypothesis[s].append(node_v)

    # # combining two tracks	
    for s,e in matches:
        for node in hypothesis[e]:
            hypothesis[s].append(node)

    # # deleting old track
    for _,e in matches:
        hypothesis[e].clear()

    return

def build_hypothesis_lst(flow_dict, source_idx, sink_idx):
    tr_end = []
    tr_bgn = []
    track_hypot = []

    for n, (k, _) in enumerate(flow_dict["source"].items()):
        tr_lst = loop_get_track(k, flow_dict)
        track_hypot.append(tr_lst)

        s_node = tr_lst[0]
        t_node = tr_lst[-1]

        if s_node[0] != source_idx:
            tr_bgn.append(n)

        if t_node[0] != sink_idx:
            tr_end.append(n)

    return track_hypot, tr_bgn, tr_end

def remove_compex_scene_id(path2hypothesis, transform, size):
    hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
    frame_indices = hypothesis[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    log_filename = './hypothesis_new.txt'
    log_file = open(log_filename, 'w')

    print ("==> removing complex scene ids")
    count = 0

    for frame_idx in range(min_frame_idx, max_frame_idx):
        rows = hypothesis[frame_indices == frame_idx]
        
        _wc = convert2world_post(rows, size, transform)

        for n,r in enumerate(rows):
            if is_patch_complex_scene(n, _wc, transform, tdist=3.5, tcrowd=6):
                count = count + 1
                continue

            tid, x1, y1, w, h, s = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[9])
            # write new result
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (frame_idx, tid, x1, y1, w, h, s))

    print (count)
    # switch all appearances of tid to new id

def remove_compex_scene_id_grid(path2hypothesis, transform, size):

    grid = {}

    v_t = 17
    h_t = 25

    gw = float(transform.parameter.get("ground_width"))
    gh = float(transform.parameter.get("ground_height"))

    x_w = gw / h_t
    y_w = gh / v_t

    x_range = [x * x_w for x in range(0, h_t)]
    y_range = [y * y_w for y in range(0, v_t)]

    for i,y in enumerate(y_range):
        ty = [y, y+y_w]
        for j,x in enumerate(x_range):
            tx = [x, x+x_w]
            pos = (i,j)
            grid[pos] = [ty,tx]

    hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
    frame_indices = hypothesis[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    log_filename = './hypothesis_new.txt'
    log_file = open(log_filename, 'w')

    print ("==> removing complex scene ids")
    b2t = {}

    for frame_idx in range(min_frame_idx, max_frame_idx):
        rows = hypothesis[frame_indices == frame_idx]
        _wc = convert2world_post(rows, size, transform)
        
        for item in grid.items():
            # item[0] -> (i,j)
            # item[1] -> [[y1 y2],[x1 x2]]
            pos = item[0]
            yran = item[1][0] # [y1 y2]
            xran = item[1][1] # [x1 x2]

            b2t[pos] = []

            for n,p in enumerate(_wc):
                # p is normalised
                xp = min(p[0]*transform.parameter.get("ground_width"), float(transform.parameter.get("ground_width")))
                yp = min(p[1]*transform.parameter.get("ground_height"), float(transform.parameter.get("ground_height")))

                if yp >= yran[0] and yp < yran[1] and xp >= xran[0] and xp < xran[1]:
                    b2t[pos].append(n)

        for item in b2t.items():
            # item[0] -> (i,j) grid pos
            # item[1] -> [] lst of indices in row

            if len(item[1]) >= 5:
                print ('pos (%d,%d) contains more than 5 obj' % (item[0][0], item[0][1]))