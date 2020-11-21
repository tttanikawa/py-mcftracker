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


# a = {'100':12,'6':5,'88':3,'test':34, '67':7,'1':64 }

def main():
	path2video = sys.argv[1]
	path2det = sys.argv[2]

	# 1. read from video
	cap = cv2.VideoCapture(path2video)
	
	# 2. read detection file
	det_in = np.loadtxt(path2det, delimiter=',')
	frame_indices = det_in[:, 0].astype(np.int)

	# Prepare initial detecton results, ground truth, and images
	# You need to change below
	# detections = {"image_name": [x1, y1, x2, y2, score]}
	# tags = {"image_name": [x1, y1, x2, y2]}
	# images = {"image_name": numpy_image}

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

		# if image_name == "1":
		# 	for i, img in enumerate(images[image_name]):
		# 		cv2.imwrite('o_%s_%d.jpg' % (image_name, i), img)

		#cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
		# cv2.imwrite("frame.jpg", frame)
		frame_num = frame_num+1
		if frame_num == 100:
			break

	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	# beta = 0.
	fib_search = False

	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit)
	tracker.build_network(images)
	optimal_flow, optimal_cost = tracker.run(fib=fib_search)
	end = time.time()

	print("Finished: {} sec".format(end - start))
	print("Optimal number of flow: {}".format(optimal_flow))
	print("Optimal cost: {}".format(optimal_cost))

	# write to file
	log_filename = './hypothesis.txt'
	log_file = open(log_filename, 'w')

	for n, (k, v) in enumerate(tracker.flow_dict.items()):
		log_file.write(str(k) + str(v) + "\n")

if __name__ == "__main__":
	main()
