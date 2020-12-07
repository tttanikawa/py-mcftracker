"""
Copyright 2018 watanika, all rights reserved.
Licensed under the MIT license <LICENSE-MIT or
http://opensource.org/licenses/MIT>. This file may
not be copied, modified,or distributed except
according to those terms.
"""

import time
from mcftracker import MinCostFlowTracker
import sys
import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment

import tools

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

def compute_cost(cur_patch, ref_patch, cur_box, ref_box, a=1., b=20, inf=1e6):
	hist1 = tools.calc_HS_histogram(cur_patch, None)
	hist2 = tools.calc_HS_histogram(ref_patch, None)
	
	color_diff = tools.calc_bhattacharyya_distance(hist1, hist2)
	distance = calc_eucl_dist(cur_box, ref_box)

	if distance > 180: 
		return inf

	cost = a*distance + b*color_diff	
	return cost

def cost_matrix(hypothesis, hypothesis_t, hypothesis_s, images, detections, inf=1e6, gap=350):
	cost_mtx = np.zeros((len(hypothesis_t), len(hypothesis_s)))

	for i, index_i in enumerate(hypothesis_t):
		for j, index_j in enumerate(hypothesis_s):
			last_idx = hypothesis[index_i][-1] # tuple ('frame_num', detection_index, 'u')
			first_idx = hypothesis[index_j][0]

			cost_mtx[i][j] = compute_cost(images[last_idx[0]][last_idx[1]], images[first_idx[0]][first_idx[1]], 
											detections[last_idx[0]][last_idx[1]], detections[first_idx[0]][first_idx[1]])

			# check if start_i > end_i
			if int(last_idx[0]) > int(first_idx[0]):
				cost_mtx[i][j] = inf
			# check if gap isn't too large
			if int(first_idx[0]) - int(last_idx[0]) > gap:
				cost_mtx[i][j] = inf

	return cost_mtx

def temporal_hungarian_matching(hypothesis, hypothesis_t, hypothesis_s, images, detections):
	cost = cost_matrix(hypothesis, hypothesis_t, hypothesis_s, images, detections)
	
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

def main(path2video, path2det, start_frame, stop_frame):	
	# 2. read detection file
	det_in = np.loadtxt(path2det, delimiter=',')
	frame_indices = det_in[:, 0].astype(np.int)

	detections = {}
	tags = {}
	images = {}
	
	print ('# starting to read input frames & detection data')

	vidcap = cv2.VideoCapture(path2video)
	success = True

	while success and vidcap.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
		success, frame = vidcap.read()

	frame_num = start_frame

	while success and vidcap.get(cv2.CAP_PROP_POS_FRAMES) <= stop_frame:
		success, frame = vidcap.read()
		
		if frame_num == 0:
			frame_num = frame_num + 1
			continue

		if frame_num % 100 == 0:
			print ('-> reading frame #%d' % (frame_num))

		mask = frame_indices == frame_num
		rows = det_in[mask]

		image_name = "%d" % (frame_num)

		bboxes = []
		bbtags = []
		bbimgs = []

		for r in rows:
			_, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
			imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]

			# filtering scores less than .51, allowing vals less than .51 causes bug in graph
			if s < 0.51:
				continue

			bbimgs.append( imgbox )
			bboxes.append( [x1,y1,x2,y2,s] )
			bbtags.append( [x1,y1,x2,y2] )

		# 3. fill in dictionaries
		detections[image_name] = bboxes
		tags[image_name] = bbtags
		images[image_name] = bbimgs

		frame_num = frame_num+1
	
	print ('# starting to execute main algorithm')
	
	vidcap.release()

	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	fib_search = False

	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit)
	print ('-> start building min cost flow graph')
	tracker.build_network(images, str(frame_num-1))
	print ('-> finish building min cost flow graph')
	optimal_flow, optimal_cost = tracker.run(fib=fib_search)
	end = time.time()

	print("Finished: {} sec".format(end - start))
	print("Optimal number of flow: {}".format(optimal_flow))
	print("Optimal cost: {}".format(optimal_cost))

	# write to file
	log_filename = './hypothesis.txt'
	log_file = open(log_filename, 'w')
	
	tr_end = []
	tr_bgn = []

	track_hypot = []
	
	start_offset = str(start_frame+1) if start_frame == 0 else str(start_frame) 
	end_offset = str(stop_frame)

	print ('-> offset interval [%s-%s]' % (start_offset, end_offset))

	for n, (k,_) in enumerate(tracker.flow_dict["source"].items()):
		tr_lst = loop_get_track(k, tracker.flow_dict)
		track_hypot.append(tr_lst)

		s_node = tr_lst[0]
		t_node = tr_lst[-1]

		if s_node[0] != start_offset:
			tr_bgn.append(n)

		if t_node[0] != end_offset:
			tr_end.append(n)

	print ('tracks not finished at sink')
	for index in tr_end:
		print('track index %d finished at frame %s' % (index+1, track_hypot[index][-1]))

	print ('tracks not started at source')
	for index in tr_bgn:
		print('track index %d started at frame %s' % (index+1, track_hypot[index][0]))

	temporal_hungarian_matching(track_hypot, tr_end, tr_bgn, images, detections)

	for id, track in enumerate(track_hypot):
		mf = 1
		for i, t in enumerate(track):
			if i % 2 == 0:
				bi = int(t[1])
				b = detections[t[0]][bi]
				f = int(t[0])

				l = str(f) + "," + str(mf*(id+1)) + "," + str(b[0]) + "," + str(b[1]) + "," + str(b[2]) + "," + str(b[3]) + "\n"
				log_file.write(l)
	return

def visualise_hypothesis(video, path2det, num_skip_frames):
	hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
	detections = np.loadtxt(path2det, delimiter=',')

	frame_indices = hypothesis[:, 0].astype(np.int)
	frame_indices_dets = detections[:, 0].astype(np.int)

	min_frame_idx = frame_indices.astype(np.int).min()
	max_frame_idx = frame_indices.astype(np.int).max()
	
	out_size = (2400, 600)
	vout = cv2.VideoWriter('./out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, out_size)

	cap = cv2.VideoCapture(video)

	success = True
	while success and cap.get(cv2.CAP_PROP_POS_FRAMES) < num_skip_frames:
		success, frame = cap.read()

	for frame_idx in range(min_frame_idx, max_frame_idx + 1):

		if frame_idx % 100 == 0:
			print("Frame %05d/%05d" % (frame_idx, max_frame_idx))

		mask_h = frame_indices == frame_idx
		mask_d = frame_indices_dets == frame_idx
		
		rows = hypothesis[mask_h]
		dets = detections[mask_d]

		_, frame = cap.read()

		cv2.putText(frame, str(frame_idx), (150, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4)

		for r in rows:	
			tid, x1, y1, x2, y2 = int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])
			cv2.putText(frame, str(tid), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
		
		for d in dets:	
			_, x1, y1, x2, y2,_ = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), int(d[5])
			cv2.circle(frame, (x1+int((x2-x1)/2), y1+int((y2-y1)/2)), 2, (255,0,0), 5)

		vout.write(cv2.resize(frame, out_size))

	cap.release()
	vout.release()

if __name__ == "__main__":
	path2video = sys.argv[1]
	path2det = sys.argv[2]

	start_f = int(sys.argv[3])
	end_f = int(sys.argv[4])

	main(path2video, path2det, start_f, end_f)
	visualise_hypothesis(path2video, path2det, start_f)
