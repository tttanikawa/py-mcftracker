import tools
from scipy.optimize import linear_sum_assignment
import numpy as np

import sys

import torch

from bepy.transform import Transform
from bepy.models import MatchVideo

import cv2
from scipy.spatial import distance

import debug
import copy
import utils

import matplotlib.pyplot as plt
from scipy.misc import face

import math

from node import GraphNode
from scipy import interpolate
from bbox import Box

from pfe.torchreid.utils import FeatureExtractor
from pfe.scripts.extract_fetures import network_feed_from_list

sys.path.append("../")
from maskrcnn_mmdet.demo.inference_mask import init_segmentor, get_masks_from_image_lst

import mmcv

import KalmanFilter
from Tracker import OnlineTracker

def isGoalArea(transform, xwc, size=None, frame=None):

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

def _test_tracker_online(path2det, path2video, slice_start, slice_end, det_in, frame_indices, match_video_id, 
                    out_file, min_confidence=0.4, max_iou=0.98):

    video = mmcv.VideoReader(path2video)
    match_video = MatchVideo.load(match_video_id)

    transform = Transform(
        match_video.video.camera_recording["parameter"],
        match_video.video.camera_recording["extrinsic_json"],
        match_video.video.camera_recording["stitching_json"],
    )

    parity = False

    if slice_start %2 != 0 and (slice_end-1) % 2 != 0: # odd - odd
        parity = True
    elif slice_start %2 == 0 and (slice_end-1) % 2 == 0: # even - even
        parity = True
    
    tracker = OnlineTracker()

    fnum = 0
    all_lines = []
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

        if fnum % 10 == 0:
            print ('-> reading frame %d / %d' % (fnum, slice_end-slice_start))

        mask = frame_indices == (index - slice_start + 1)
        rows = det_in[mask]

        boxes = []

        for _,r in enumerate(rows):
            _, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            boxes.append(Box([x1,y1,x2,y2], s, transform, size))

        boxes = [b for b in boxes if b.confidence >= min_confidence]

        # Run non-maxima suppression.
        blst = np.array([b.to_tlwh() for b in boxes])
        scrlst = np.array([b.confidence for b in boxes])
        indices = utils.non_max_suppression(blst, max_iou, scrlst)
        boxes_nms = [boxes[i] for i in indices]

        if fnum == 1:
            tracker.onlineTrackerInit(boxes_nms)
            continue
 
        matches, unmatched_track_ids, unmatched_det_ids = tracker.onlineTrackerAssign(boxes_nms, index-slice_start+1)

        # for t in unmatched_track_ids:
        #     print ('[%d] unmatched track id %d' % (index-slice_start+1, tracker.tracks[t].id))

        # for d in unmatched_det_ids:
        #     print ('[%d] unmatched det idx %d'%(index-slice_start+1, d))

        lines = []
        for match in matches:
            tidx, didx = match[0], match[1]
            box, track = boxes_nms[didx], tracker.tracks[tidx]
            l = [index-slice_start+1, track.id, box.tlbr[0], box.tlbr[1], box.tlbr[2]-box.tlbr[0], box.tlbr[3]-box.tlbr[1], 0.99]
            lines.append(l)

        all_lines.append(lines)

        tracker.onlineTrackerUpdate(matches, boxes_nms)

    interpolate_lines(out_file, all_lines, parity)
    
    return [], transform, size

def read_input_data(path2det, path2video, slice_start, slice_end, det_in, frame_indices, match_video_id,
                        # min_confidence=0.4, max_iou=0.98, segment=False, ckpt_path='/root/py-mcftracker/pfe/checkpoints/market_combined_120e.pth'):
                        min_confidence=0.4, max_iou=0.98, segment=False, ckpt_path='/home/bepro/py-mcftracker/pfe/checkpoints/market_combined_120e.pth'):
    
    input_data = {}
    video = mmcv.VideoReader(path2video)

    if segment:
        segmentor = init_segmentor()
    else:
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
    
    last_frame = ""
    wc_d = {}

    # online tracker to extract motion model
    tracker = OnlineTracker()

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

        if fnum % 100 == 0:
            print ('-> reading frame %d / %d' % (fnum, slice_end-slice_start))

        mask = frame_indices == (index - slice_start + 1)
        rows = det_in[mask]

        image_name = "%d" % (fnum)
        last_frame = image_name

        bbimgs = []
        node_lst = []
        _wc = []
        boxes = []

        for n,r in enumerate(rows):
            _, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            boxes.append(Box([x1,y1,x2,y2], s, transform, size))

        boxes = [b for b in boxes if b.confidence >= min_confidence]

        # Run non-maxima suppression.
        blst = np.array([b.to_tlwh() for b in boxes])
        scrlst = np.array([b.confidence for b in boxes])
        indices = utils.non_max_suppression(blst, max_iou, scrlst)
        boxes_nms = [boxes[i] for i in indices]

        for box in boxes_nms:
            imgbox = frame[int(box.tlbr[1]):int(box.tlbr[3]), int(box.tlbr[0]):int(box.tlbr[2]), :]
            p = box.to_world()
            _wc.append(p)

            mskt_ = np.full((imgbox.shape[0],imgbox.shape[1]), True, dtype=bool)
            node = GraphNode(p, box.tlbr, box.confidence, 0, mask=mskt_)

            bbimgs.append(imgbox)
            node_lst.append(node)
        
        wc_d[index] = _wc.copy()

        if len(bbimgs) == 0:
            print ('no images')
            continue

        if segment:    
            masks = get_masks_from_image_lst(segmentor, bbimgs)
            masks_p = utils.preprocess_masks(masks, bbimgs)

            for m in masks_p:
                idx = m[0]
                mask = m[1]
                node_lst[idx]._mask = mask
                    
            for i,node in enumerate(node_lst):
                node._hist = tools.calc_RGB_histogram(bbimgs[i], node._mask)

        else:
            feats = network_feed_from_list(bbimgs, extractor)
            for i,node in enumerate(node_lst):
                node._feat = feats[i]

        if fnum == 1:
            tracker.onlineTrackerInit(boxes_nms)
        else:
            matches, _, _ = tracker.onlineTrackerAssign(boxes_nms, index-slice_start+1)

            for match in matches:
                tidx, didx = match[0], match[1]
                dnode, track = node_lst[didx], tracker.tracks[tidx]
                
                dnode._observed = True
                dnode._kf = track.kf
                dnode._mean = track.mean
                dnode._covar = track.covariance

            tracker.onlineTrackerUpdate(matches, boxes_nms)

        input_data[image_name] = node_lst

    print ('-> %d images have been read & processed' % (len(input_data)))

    return input_data, transform, size, parity, wc_d, last_frame

def interpolate_lines(log_filename, all_lines, parity, draw_mask=False):
    # write to file
    log_file = open(log_filename, 'w')

    if draw_mask:
        log_file_mask = open('./masks.txt', 'w')

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
                if draw_mask:
                    log_file_mask.write(str(l[7])+'\n')

            for l in al:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))
                if draw_mask:
                    log_file_mask.write(str(l[7])+'\n')
        
        # last index
        li = len(all_lines)-1
        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))
            if draw_mask:
                log_file_mask.write(str(l[7])+'\n')

        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0]+1, l[1], l[2], l[3], l[4], l[5], l[6]))
            if draw_mask:
                log_file_mask.write(str(l[7])+'\n')

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
                if draw_mask:
                    log_file_mask.write(str(l[7])+'\n')

            for l in al:
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))
                if draw_mask:
                    log_file_mask.write(str(l[7])+'\n')

        # write last line
        li = len(all_lines)-1
        for l in all_lines[li]:
            log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (l[0], l[1], l[2], l[3], l[4], l[5], l[6]))
            if draw_mask:
                log_file_mask.write(str(l[7])+'\n')

    return 

def write_output_data(log_filename, track_hypot, path2det, data, iend, frame_offset, iid, parity, draw_mask=True):

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

                        if draw_mask:
                            wc = data[t[0]][bi]._3dc
                            if wc[0] == 0. and wc[1] == 0.: # interpolated in tracklet matching
                                mask = [[]]
                            else:
                                mask = data[t[0]][bi]._mask.astype(np.uint8).tolist()
                            lines.append([f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], s, mask])
                        else:
                            lines.append([f, (iid-1)*10000+(id+1), b[0], b[1], b[2]-b[0], b[3]-b[1], s])

        all_lines.append(lines)
        f = f+2
    
    interpolate_lines(log_filename, all_lines, parity)

    return

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

def build_hypothesis_lst(flow_dict, source_idx, sink_idx):
    track_hypot = []

    nh = [] # no head
    nt = [] # no tail
    nht = [] # no head no tail
    ht = []

    print ('source frame %s sink frame %s' % (source_idx, sink_idx))

    for n, (k, _) in enumerate(flow_dict["source"].items()):
        tr_lst = loop_get_track(k, flow_dict)
        track_hypot.append(tr_lst)

        s_node = tr_lst[0]
        t_node = tr_lst[-1]

        if s_node[0] != source_idx and t_node[0] == sink_idx:
            nh.append(n)
        elif s_node[0] == source_idx and t_node[0] != sink_idx:
            nt.append(n)
        elif s_node[0] != source_idx and t_node[0] != sink_idx:
            nht.append(n)
        elif s_node[0] == source_idx and t_node[0] == sink_idx:
            ht.append(n)
                
    print ('# tracklets not ending at last frame: %d' % (len(nt)))
    print ('# tracklets not starting at first frame: %d' % (len(nh)))
    print ('# tracklets not starting at first and not ending at last frame: %d' %(len(nht)))
    print ('# tracklets complete: %d' % (len(ht)))

    return track_hypot, nt, nh, nht

def remove_compex_scene_id(path2hypothesis, transform, size):
    hypothesis = np.loadtxt("./hypothesis_.txt", delimiter=',')
    frame_indices = hypothesis[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    log_filename = './hypothesis.txt'
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

def remove_compex_scene_id_grid(file_in, file_out, transform, size):
    v_t = 20
    h_t = 30

    gw = float(transform.parameter.get("ground_width"))
    gh = float(transform.parameter.get("ground_height"))

    x_w = gw / h_t
    y_w = gh / v_t

    x_range = [x * x_w for x in range(0, h_t)]
    y_range = [y * y_w for y in range(0, v_t)]

    grid = {}
    for i,y in enumerate(y_range):
        ty = [y, y+y_w]
        for j,x in enumerate(x_range):
            tx = [x, x+x_w]
            pos = (i,j)
            grid[pos] = [ty,tx]

    hypothesis = np.loadtxt(file_in, delimiter=',')
    frame_indices = hypothesis[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    # log_filename = './hypothesis.txt'
    log_file = open(file_out, 'w')

    print ("==> removing complex scene ids")
    b2t = {}

    for frame_idx in range(min_frame_idx, max_frame_idx+1):
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
            if len(item[1]) < 4:
                for i in item[1]:
                    r = rows[i]
                    tid, x1, y1, w, h, s = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[9])
                    # write new result
                    log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (frame_idx, tid, x1, y1, w, h, s))

        # writing id's outside of pitch
        for n,p in enumerate(_wc):
            # p is normalised
            xp = p[0]*transform.parameter.get("ground_width")
            yp = p[1]*transform.parameter.get("ground_height")

            if xp<=0. or yp<=0. or xp>=float(transform.parameter.get("ground_width")) or yp>=float(transform.parameter.get("ground_height")):
                r = rows[n]
                tid, x1, y1, w, h, s = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[9])
                # write new result
                log_file.write('%d, %d, %f, %f, %f, %f, 1,-1,-1, %d \n' % (frame_idx, tid, x1, y1, w, h, s))


def interpolate_gap(tracks, data):
    
    for j,track in enumerate(tracks):
        # track -> [('1', 24, 'u'), ... ]
        new_track = []

        new_track.append(track[0])
        new_track.append(track[1])

        for i in range(2, len(track), 2):
            tpre = track[i-2]
            tcur = track[i]

            new_track.append(track[i])
            new_track.append(track[i+1])

            fpre = int(tpre[0])
            fcur = int(tcur[0])

            if (fcur-fpre)>1 and (fcur-fpre)<=20:
                bs = data[tpre[0]][tpre[1]]._bb
                be = data[tcur[0]][tcur[1]]._bb

                # print ('%d -> %d > %s -> %s' % (fpre,fcur,bs,be))

                fpx1 = [bs[0], be[0]]
                fpy1 = [bs[1], be[1]]
                fpx2 = [bs[2], be[2]]
                fpy2 = [bs[3], be[3]]

                xp = [fpre, fcur]

                for x in range(fpre+1, fcur):
                    pix1 = np.interp(x, xp, fpx1)
                    piy1 = np.interp(x, xp, fpy1)
                    pix2 = np.interp(x, xp, fpx2)
                    piy2 = np.interp(x, xp, fpy2)

                    # print ('%d -> %s' % (x, [pix1,piy1,pix2,piy2]))
                    pi = [pix1,piy1,pix2,piy2]

                    gn = GraphNode([0.,0.], pi, 0., 0)
                    fn = str(x)

                    index = len(data[fn])
                    data[fn].append(gn)

                    node_u = (fn, index, "u")
                    node_v = (fn, index, "v")

                    new_track.append(node_u)
                    new_track.append(node_v)

        # sort new_track by fn
        new_track_s = sorted(new_track, key=lambda x: int(x[0]))
        # assign new_track
        tracks[j] = new_track_s    