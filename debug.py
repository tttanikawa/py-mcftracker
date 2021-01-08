import tools
import cv2
from scipy.ndimage import rotate
import numpy as np
import mmcv
from random import randrange
from scipy.spatial import distance
import operator

def get_patch_by_id(tracker, track_num, detections, images, features):
    # get hypothesis file
    # extract all lines with field == track_num
    frame_infos = []
    with open('./hypothesis.txt') as fp:
        for line in fp:
            line_lst = line.split(',')

            if int(line_lst[1]) == track_num:
                frame_infos.append( ( int(line_lst[0]), float(line_lst[2]), float(line_lst[3]) ) )

    det_indices = []
    for (frame_num, x, y) in frame_infos:
        for i,d in enumerate(detections[str(frame_num)]):
            if d[0] == x and d[1] == y:
                det_indices.append((frame_num, i))
                break

    # save link costs 
    for index in range(1, len(det_indices)):
        prev_det_data =  det_indices[index-1]
        prev_img_name = str(prev_det_data[0])
        prev_det_idx = prev_det_data[1]

        cur_det_data = det_indices[index]
        cur_img_name = str(cur_det_data[0])
        cur_det_idx = cur_det_data[1]

        cd = distance.cosine(features[prev_img_name][prev_det_idx], features[cur_img_name][cur_det_idx])

        if cur_img_name == "2515":
            cv2.imwrite ('./%s_orig_patch_cur.jpg' % (cur_img_name), images[cur_img_name][cur_det_idx])
            cv2.imwrite ('./%s_orig_patch_prev.jpg' % (prev_img_name), images[prev_img_name][prev_det_idx])

        rot_img = rotate(images[cur_img_name][cur_det_idx], 90)
        h = rot_img.shape[0]
        w = rot_img.shape[1]

        cv2.putText(rot_img, str(round(cd, 6)), (1, int(h/2)+3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        out_img = rotate(rot_img, -90)
        cv2.imwrite('./id_%d_f_%s.jpg' % (track_num, cur_img_name), out_img)


def visualise_hypothesis(path2video, path2det, frame_offset, frame_count):

    hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
    detections = np.loadtxt(path2det, delimiter=',')

    frame_indices = hypothesis[:, 0].astype(np.int)
    frame_indices_dets = detections[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    
    out_size = (1800, 550)
    vout = cv2.VideoWriter('./out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, out_size)

    video = mmcv.VideoReader(path2video)

    slice_start = 0 if frame_offset == 0 else frame_offset-1
    slice_end = min(frame_offset+frame_count, frame_offset+max_frame_idx)

    for frame_idx in range(slice_start, slice_end):
        frame = video[frame_idx]
        
        if (frame_idx+1) % 500 == 0:
            print("Frame %05d/%05d" % (frame_idx+1, slice_end))

        mask_h = frame_indices == (frame_idx - slice_start + 1)
        mask_d = frame_indices_dets == (frame_idx - slice_start + 1)
        
        rows = hypothesis[mask_h]
        dets = detections[mask_d]

        cv2.putText(frame, str(frame_idx+1), (150, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4)

        for r in rows:	
            tid, x1, y1, w, h = int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])
            cv2.putText(frame, str(tid), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        
        for d in dets:	
            _, x1, y1, x2, y2,_ = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), int(d[5])
            cv2.circle(frame, (x1+int((x2-x1)/2), y1+int((y2-y1)/2)), 2, (255,0,0), 5)

        cv2.putText(frame, '# dets ' + str(len(dets)), (150, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 4)

        vout.write(cv2.resize(frame, out_size))

    vout.release()

def validate_appereance_model(image, detections, features):
    n = len(detections)
    anchor_idx = randrange(n-1)

    x = {}

    for i, det in enumerate(detections):

        if i == anchor_idx:
            continue

        cos_diff = distance.cosine(features[anchor_idx], features[i])
        # x[det] = cos_diff
        x[i] = cos_diff

    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    group_blue = []

    for item in sorted_x:
        if item[1] <= 0.400:
            group_blue.append((detections[item[0]], item[1]))

    col = 255, 0, 0
    _col = 0, 255, 0
    thick = 3

    for ind, tup in enumerate(group_blue):
        bbox, score = tup[0], tup[1]
        pt1 = int(bbox[0]), int(bbox[1])
        pt2 = int(bbox[2]), int(bbox[3])
        center = pt1[0] + 5, pt1[1] + 5
        cv2.rectangle(image, pt1, pt2, col, thick)
        # cv2.putText(image, str(det.confidence), center, cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3)
        cv2.putText(image, str(round(score, 6)), center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
    
    _pt1 = int(detections[anchor_idx][0]), int(detections[anchor_idx][1]) 
    _pt2 = int(detections[anchor_idx][2]), int(detections[anchor_idx][3]) 

    cv2.rectangle(image, _pt1, _pt2, _col, thick)
    cv2.imwrite('appearance_eval.jpg', image)


def return_closest_box_index(cur_box, ref_frame_dets, frame, cur_box_index):

    max_iou = 0.
    max_index = -1

    for j, det in enumerate(ref_frame_dets):

        iou = tools.calc_overlap(cur_box, det)

        if iou > max_iou:
            max_iou = iou
            max_index = j

    if max_index == -1:
        print ("[DBG] no nearby box for box index: %d in the previous frame" % (cur_box_index))
        cv2.rectangle(frame, (int(cur_box[0]), int(cur_box[1])), (int(cur_box[2]), int(cur_box[3])), (0, 0, 255), 3)
        return -1

    return max_index

def validate_cosine_with_detections(path2video, frame_num_lst, detections, features):
    # frame_num - frame number of anchor frame: cos distance between this frame and previous will be visualised

    for frame_num in frame_num_lst:
        video = mmcv.VideoReader(path2video)
        frame = video[frame_num-1]

        image_name = str(frame_num)
        prev_img_name = str(frame_num-1)

        print ('f %d' % (frame_num))
        print ('# of features prev %d cur %d' % (len(features[image_name]), len(features[prev_img_name])))
        print ('# of detections prev %d cur %d' % (len(detections[image_name]), len(detections[prev_img_name])))

        for i, det in enumerate(detections[image_name]):
            ref_index = return_closest_box_index(det, detections[prev_img_name], frame, i)
            d = distance.cosine(features[image_name][i], features[prev_img_name][ref_index])

            pt1 = int(det[0]), int(det[1])
            pt2 = int(det[2]), int(det[3])
            center = pt1[0] + 5, pt1[1] + 5
            cv2.putText(frame, str(round(d, 6)), center, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

        cv2.imwrite('./%d_validate_cosine_with_dets.jpg' % (frame_num), frame)


