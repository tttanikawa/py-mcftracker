import tools
import cv2
from scipy.ndimage import rotate
import numpy as np
import mmcv

def get_patch_by_id(tracker, track_num, detections, images):
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

		# lc = tracker._calc_cost_link(detections[prev_img_name][prev_det_idx], detections[cur_img_name][cur_det_idx],
		# 									images[prev_img_name][prev_det_idx], images[cur_img_name][cur_det_idx], False)

		h1 = tools.calc_HS_histogram(images[prev_img_name][prev_det_idx])
		h2 = tools.calc_HS_histogram(images[cur_img_name][cur_det_idx])
		pc = 1.0 - tools.calc_bhattacharyya_distance(h1, h2)

		rot_img = rotate(images[cur_img_name][cur_det_idx], 90)
		h = rot_img.shape[0]
		w = rot_img.shape[1]

		# cv2.putText(rot_img, str(lc), (1, int(h/2)+3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
		cv2.putText(rot_img, str(pc), (1, int(h/2)+3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
		out_img = rotate(rot_img, -90)
		cv2.imwrite('./id_%d_f_%s.jpg' % (track_num, cur_img_name), out_img)


def visualise_hypothesis(path2video, path2det, frame_offset, frame_count):

    hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
    detections = np.loadtxt(path2det, delimiter=',')

    frame_indices = hypothesis[:, 0].astype(np.int)
    frame_indices_dets = detections[:, 0].astype(np.int)

    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    
    out_size = (1800, 600)
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