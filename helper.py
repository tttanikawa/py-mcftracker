import tools
from scipy.optimize import linear_sum_assignment
import numpy as np
import mmcv

import sys
sys.path.append('/root/py-mcftracker/player-feature-extractor')
sys.path.append('/root/bepro-python')

import torch
from torchreid.utils import FeatureExtractor
from scripts.extract_fetures import network_feed_from_list

from bepy.transform import Transform
from bepy.models import MatchVideo

import cv2
from scipy.spatial import distance

import debug
import copy

import matplotlib.pyplot as plt
from scipy.misc import face

import math

def isComplexArea(transform, xwc, size=None, frame=None):

    gp = [[0, 54.16, 0], [16.5, 54.16, 0], [0, 13.84, 0], [16.5, 13.84, 0],
            [105, 54.16, 0], [88.5, 54.16, 0], [105, 13.84, 0], [88.5, 13.84, 0]]

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

def is_patch_reliable(tlbr, boxes):
    # calculate iou with all boxes in current frame
    for box in boxes:
        _, x1, y1, x2, y2, s = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5])
        cb = [x1,y1,x2,y2,s]
        
        if cb == tlbr:
            continue

        if tools.calc_overlap(tlbr[:4], cb[:4]) > 1.0:
        # if tools.calc_overlap(tlbr[:4], cb[:4]) > 0.1:
            return False

    return True

def is_patch_complex_scene(index, wc, transform, size=None, image=None, tdist=4.0, tcrowd=6):
    crowd = 0
    cb = wc[index]
    cx, cy = cb[0], cb[1]

    if not isComplexArea(transform, cb, size, image):
        return False

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
        wc.append((cx,cy))
    return wc

def read_input_data(path2det, path2video, slice_start, slice_end, det_in, frame_indices, match_video_id,
                        ckpt_path='/root/py-mcftracker/player-feature-extractor/checkpoints/market_combined_120e.pth'):
    
    detections = {}
    tags = {}
    images = {}
    features = {}

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

    for index in range(slice_start, slice_end):
        frame = video[index]

        if index == slice_start:
            size = frame.shape
            cv2.imwrite("./frame.jpg", frame)

        if (index+1) % 500 == 0:
            print ('-> reading frame %d / %d' % (index+1, slice_end))

        mask = frame_indices == (index - slice_start + 1)
        rows = det_in[mask]

        image_name = "%d" % (index+1)

        bboxes = []
        bbtags = []
        bbimgs = []

        _wc = convert2world(rows, size, transform)

        for n,r in enumerate(rows):
            _, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])

            curbox = [x1,y1,x2,y2,s]
            imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]

            if not is_patch_complex_scene(n, _wc, transform, size, frame):
                # check complex scene: if complex scene skip
                bbimgs.append( imgbox )
                bboxes.append( curbox )
                bbtags.append( [x1,y1,x2,y2] )

        if len(bbimgs) == 0:
            print ('no images')
            continue

        feats = network_feed_from_list(bbimgs, extractor)

        # 3. fill in dictionaries
        detections[image_name] = copy.deepcopy(bboxes)
        tags[image_name] = copy.deepcopy(bbtags)
        images[image_name] = copy.deepcopy(bbimgs)
        features[image_name] = copy.deepcopy(feats)

    print ('-> %d images have been read & processed' % (len(detections)))

    return detections, tags, images, features, transform, size

def write_output_data(track_hypot, path2det, detections, slice_start, slice_end, frame_offset, iid):
    # write to file
    log_filename = './hypothesis.txt'
    log_file = open(log_filename, 'w')

    for n in range(slice_start+1, slice_end+1):
        for id, track in enumerate(track_hypot):
            for i, t in enumerate(track):
                if i % 2 == 0:
                    
                    if int(t[0]) == n:
                        bi = int(t[1])
                        b = detections[t[0]][bi]
                        # f = int(t[0]) - frame_offset
                        f = int(t[0]) if frame_offset == 0 else int(t[0]) - frame_offset + 1
                        
                        # must be in top-left-width-height
                        # log_file.write('%d, %d, %.2f, %.2f, %.2f, %.2f, 1,-1,-1, %d \n' % (f, (iid-1)*10000+(id+1), b[0], b[1], b[2], b[3], 1))
                        # log_file.write('%d, %d, %.2f, %.2f, %.2f, %.2f, 1,-1,-1, %d \n' % (f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], 1))
                        log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], 1))

def extract_patch_block(patch):
    # image.shape [h, w, c]
    # split in 3 of height
    # imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]
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
    return math.log(x) + 2.3

def compute_cost(u, v, cur_box, ref_box, transform, size, frame_gap, alpha=0.8, inf=1e6):

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

    if not isComplexArea(transform, (rx,ry)):
        return inf

    # print (transform.parameter.get("ground_width"), transform.parameter.get("ground_height"))

    dist = calc_eucl_dist([cx*transform.parameter.get("ground_width"),cy*transform.parameter.get("ground_height")], 
                        [rx*transform.parameter.get("ground_width"),ry*transform.parameter.get("ground_height")])

    maxdistance = return_max_dist(frame_gap)

    if dist > maxdistance: 
        return inf

    dist_norm = dist / maxdistance
    cost = alpha*dist_norm + (1-alpha)*cos_dist

    return cost

def cost_matrix(hypothesis, hypothesis_t, hypothesis_s, features, detections, transform, size, inf=1e6, max_gap=100):
    cost_mtx = np.zeros((len(hypothesis_t), len(hypothesis_s)))

    for i, index_i in enumerate(hypothesis_t):
        for j, index_j in enumerate(hypothesis_s):
            last_idx = hypothesis[index_i][-1] # tuple ('frame_num', detection_index, 'u')
            first_idx = hypothesis[index_j][0]

            gap = int(first_idx[0]) - int(last_idx[0])

            cost_mtx[i][j] = compute_cost(features[last_idx[0]][last_idx[1]], features[first_idx[0]][first_idx[1]], 
                                            detections[last_idx[0]][last_idx[1]], detections[first_idx[0]][first_idx[1]],
                                            transform, size, gap)

            # check if gap isn't too large
            if gap >= max_gap:
                cost_mtx[i][j] = inf

    return cost_mtx

def temporal_hungarian_matching(hypothesis, hypothesis_t, hypothesis_s, features, detections, transform, size, match_video_id=57824):
    cost = cost_matrix(hypothesis, hypothesis_t, hypothesis_s, features, detections, transform, size)
    
    # assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    print ('temporal hungarian assignment')

    matches = []
    for r,c in zip(row_ind, col_ind):
        if cost[r][c] != 1e6:
            print ('id %d -> id %d - frames: [%d-%d] cost: %f' % (hypothesis_t[r]+1, hypothesis_s[c]+1, 
                        int(hypothesis[hypothesis_t[r]][-1][0]), int(hypothesis[hypothesis_s[c]][0][0]), cost[r][c]))
            matches.append((hypothesis_t[r], hypothesis_s[c]))

    print ('matches before sorting ==>')
    print (matches)
    # sort matches in descending order of hypothesis_s
    matches.sort(key=lambda tup: tup[1], reverse=True)
    print ('matches after sorting ==>')
    print (matches)

    # combining two tracks	
    for s,e in matches:
        for node in hypothesis[e]:
            hypothesis[s].append(node)

    # deleting old track
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

