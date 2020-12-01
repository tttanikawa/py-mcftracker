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
		elem = dct[elem].items()[0][0]
	return lst

def main(path2video, path2det, max_frame):
	# 1. read from video
	cap = cv2.VideoCapture(path2video)
	
	# 2. read detection file
	det_in = np.loadtxt(path2det, delimiter=',')
	frame_indices = det_in[:, 0].astype(np.int)

	detections = {}
	tags = {}
	images = {}
	frame_num = 0
	
	print ('# starting to read input frames & detection data')

	while(cap.isOpened()):
		_, frame = cap.read()

		if frame_num == 0:
			frame_num = frame_num + 1
			continue

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

		if frame_num == max_frame:
			break
	
	print ('# starting to execute main algorithm')
	
	cap.release()

	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	fib_search = False

	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit)
	tracker.build_network(images, str(frame_num-1))
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
	for n, (k,_) in enumerate(tracker.flow_dict["source"].items()):
		# print (k, n)
		tr_lst = loop_get_track(k, tracker.flow_dict)
		track_hypot.append(tr_lst)

		s_node = tr_lst[0]
		t_node = tr_lst[-1]

		if s_node[0] != '1':
			tr_bgn.append(n)

		if t_node[0] != str(max_frame-1):
			tr_end.append(n)

	# print (tr_bgn)
	# print (tr_end)

	print ('tracks not finished at sink')
	for index in tr_end:
		print('track index %d finished at frame %s' % (index+1, track_hypot[index][-1]))

	print ('tracks not started at source')
	for index in tr_bgn:
		print('track index %d started at frame %s' % ((index+1)*int(track_hypot[index][0][0]), track_hypot[index][0]))
		
	for id, track in enumerate(track_hypot):
		mf = int(track[0][0])
		for i, t in enumerate(track):
			if i % 2 == 0:
				bi = int(t[1])
				b = detections[t[0]][bi]
				f = int(t[0])

				l = str(f) + "," + str(mf*(id+1)) + "," + str(b[0]) + "," + str(b[1]) + "," + str(b[2]) + "," + str(b[3]) + "\n"
				log_file.write(l)

	
	# for c in [0.1, 0.2, 0.3, 0.4, 0.7, 0.9, 0.99]:
	# 	print ('score %f -> cost det: %d' % (c, int(tracker._calc_cost_detection(c) * 10)))

def visualise_hypothesis(video, path2det):
	cap = cv2.VideoCapture(video)
	
	hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
	detections = np.loadtxt(path2det, delimiter=',')

	frame_indices = hypothesis[:, 0].astype(np.int)
	frame_indices_dets = detections[:, 0].astype(np.int)

	min_frame_idx = frame_indices.astype(np.int).min()
	max_frame_idx = frame_indices.astype(np.int).max()
	
	out_size = (2400, 600)
	vout = cv2.VideoWriter('./out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, out_size)

	for frame_idx in range(min_frame_idx, max_frame_idx + 1):
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
			# cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
		
		for d in dets:	
			_, x1, y1, x2, y2,_ = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), int(d[5])
			# cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
			cv2.circle(frame, (x1+int((x2-x1)/2), y1+int((y2-y1)/2)), 2, (0,0,255), 5)

		vout.write(cv2.resize(frame, out_size))

	cap.release()
	vout.release()

if __name__ == "__main__":
	path2video = sys.argv[1]
	path2det = sys.argv[2]
	num_frames = int(sys.argv[3])

	main(path2video, path2det, num_frames)
	visualise_hypothesis(path2video, path2det)
