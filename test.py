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
	
	track_hypot = []
	for k,v in tracker.flow_dict["source"].items():
		track_lst = []
		recursive_get_track(k, tracker.flow_dict, track_lst)
		track_hypot.append(track_lst)

	id = 0
	for track in track_hypot:
		id = id + 1
		for i, t in enumerate(track):
			if i % 2 == 0:
				bi = int(t[1])
				b = detections[t[0]][bi]
				f = int(t[0])

				l = str(f) + "," + str(id) + "," + str(b[0]) + "," + str(b[1]) + "," + str(b[2]) + "," + str(b[3]) + "\n"
				log_file.write(l)

	# for c in [0.1, 0.2, 0.3, 0.4, 0.7, 0.9, 0.99]:
	# 	print ('score %f -> cost det: %d' % (c, int(tracker._calc_cost_detection(c) * 10)))


def visualise_hypothesis(video):
	cap = cv2.VideoCapture(video)
	
	hypothesis = np.loadtxt("./hypothesis.txt", delimiter=',')
	frame_indices = hypothesis[:, 0].astype(np.int)

	min_frame_idx = frame_indices.astype(np.int).min()
	max_frame_idx = frame_indices.astype(np.int).max()
	
	# fourcc = cv2.VideoWriter_fourcc(*'H264')
	# vout = cv2.VideoWriter("./out.mp4", fourcc, 30, (frame_width, frame_height))
	# vout = cv2.VideoWriter('./out.avi', -1, 10, (frame_width,frame_height))
	# vout = cv2.VideoWriter('./out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

	out_size = (2400, 600)
	vout = cv2.VideoWriter('./out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, out_size)

	for frame_idx in range(min_frame_idx, max_frame_idx + 1):
		print("Frame %05d/%05d" % (frame_idx, max_frame_idx))

		mask = frame_indices == frame_idx
		rows = hypothesis[mask]

		_, frame = cap.read()

		cv2.putText(frame, str(frame_idx), (150, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4)

		for r in rows:
			tid, x1, y1, x2, y2 = int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])
			cv2.putText(frame, str(tid), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
			# cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
		
		vout.write(cv2.resize(frame, out_size))

	cap.release()
	vout.release()

if __name__ == "__main__":
	path2video = sys.argv[1]
	path2det = sys.argv[2]
	num_frames = int(sys.argv[3])

	main(path2video, path2det, num_frames)
	visualise_hypothesis(path2video)
