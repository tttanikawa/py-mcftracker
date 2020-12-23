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
import mmcv

from operator import itemgetter

def extract_patch_block(patch):
	# image.shape [h, w, c]
	# split in 3 of height
	# imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]
	h, w = patch.shape[0], patch.shape[1]
	half_p = patch[:int(h/2), :, ]
	s = int(0.3 * w)
	roi = half_p[:, s:w-s, :]

	return roi

def debug_get_patch_by_id(track_num, detections, images):
	# get hypothesis file
	# extract all lines with field == track_num

	frame_infos = []
	with open('./hypothesis.txt') as fp:
		for line in fp:
			line_lst = line.split(',')

			if int(line_lst[1]) == track_num:
				frame_infos.append( ( int(line_lst[0]), float(line_lst[2]), float(line_lst[3]) ) )

	# for fi in frame_infos:
	# 	print (fi)

	det_indices = []
	for (frame_num, x, y) in frame_infos:
		for i,d in enumerate(detections[str(frame_num)]):
			if d[0] == x and d[1] == y:
				det_indices.append((frame_num, i))
				break
	
	print (len(det_indices))

	for (frame_num, i) in det_indices:
		cv2.imwrite('./id_%d_f_%d.jpg' % (track_num, frame_num), images[str(frame_num)][i])

	return

def debug_print_link_cost(tracker, t1, t2, t3, fn, detections, images):

	# comparing link cost of t1 -> t3 & t2 - > t3
	# t1 -> first track number in prev frame 
	# t2 -> second track number in prev frame
	# t3 -> track number in next frame
	# fn -> prev frame number

	fi_t1 = []
	fi_t2 = []
	fi_t3 = []

	with open('./hypothesis.txt') as fp:
		for line in fp:
			line_lst = line.split(',')

			if int(line_lst[1]) == t1:
				fi_t1.append( ( int(line_lst[0]), float(line_lst[2]), float(line_lst[3]) ) )

			if int(line_lst[1]) == t2:
				fi_t2.append( ( int(line_lst[0]), float(line_lst[2]), float(line_lst[3]) ) )

			if int(line_lst[1]) == t3:
				fi_t3.append( ( int(line_lst[0]), float(line_lst[2]), float(line_lst[3]) ) )


	det_indices = []
	for (frame_num, x, y) in fi_t1:
		if frame_num == fn:
			for i,d in enumerate(detections[str(frame_num)]):
				if d[0] == x and d[1] == y:
					det_indices.append(i)
					break
	
	for (frame_num, x, y) in fi_t2:
		if frame_num == fn:
			for i,d in enumerate(detections[str(frame_num)]):
				if d[0] == x and d[1] == y:
					det_indices.append(i)
					break

	for (frame_num, x, y) in fi_t3:
		if frame_num == fn+1:
			for i,d in enumerate(detections[str(frame_num)]):
				if d[0] == x and d[1] == y:
					det_indices.append(i)
					break

	prev_img_name = str(fn)
	cur_img_name = str(fn+1)

	diff_t1 = tracker._calc_cost_link(tracker._detections[prev_img_name][det_indices[2]], tracker._detections[cur_img_name][det_indices[0]],
										images[prev_img_name][det_indices[2]], images[cur_img_name][det_indices[0]], False)

	diff_t2 = tracker._calc_cost_link(tracker._detections[prev_img_name][det_indices[2]], tracker._detections[cur_img_name][det_indices[1]],
										images[prev_img_name][det_indices[2]], images[cur_img_name][det_indices[1]], False)


	print ('cost of transition of track %d -> track %d = %f' % (t1, t3, diff_t1))
	print ('cost of transition of track %d -> track %d = %f' % (t2, t3, diff_t2))

	cv2.imwrite('./id_%d_f_%d.jpg' % (t1, fn), images[str(fn)][det_indices[0]])
	cv2.imwrite('./id_%d_f_%d.jpg' % (t2, fn), images[str(fn)][det_indices[1]])
	cv2.imwrite('./id_%d_f_%d.jpg' % (t3, fn+1), images[str(fn+1)][det_indices[2]])

	return


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

def is_patch_reliable(tlbr, boxes):
	# calculate iou with all boxes in current frame
	for box in boxes:
		_, x1, y1, x2, y2, s = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5])
		cb = [x1,y1,x2,y2,s]
		
		if cb == tlbr:
			continue

		if tools.calc_overlap(tlbr[:4], cb[:4]) > 0.1:
			return False

	return True
	
def find_prev_imgbox(box_cur, detections, images, name, frame):
	prev_index = str(int(name)-1)

	if prev_index not in detections:
		# print (prev_index)
		# print ('[bug] Unreliable box found in the first frame of the chunk')
		# return frame[int(y1):int(y2), int(x1):int(x2), :]
		return None, False
		# sys.exit()
 
	max_iou_index = -1
	max_iou = 0.

	for n, box in enumerate(detections[prev_index]):
		# x1, y1, x2, y2, s = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
		iou = tools.calc_overlap(box_cur, box[:4])
		
		if iou > max_iou:
			max_iou = iou
			max_iou_index = n

	if max_iou_index == -1:
		# print (name, prev_index)
		# print ('[bug] No nearby box in previous frame was found!')
		return None, False
		# sys.exit()

	return images[prev_index][max_iou_index], True

def main(path2video, path2det, frame_offset, frame_count, iid):	
	detections = {}
	tags = {}
	images = {}
	
	print ('# starting to read input frames & detection data')

	det_in = np.loadtxt(path2det, delimiter=',')
	frame_indices = det_in[:, 0].astype(np.int)
	min_frame_idx = frame_indices.astype(np.int).min()
	max_frame_idx = frame_indices.astype(np.int).max()

	slice_start = 0 if frame_offset == 0 else frame_offset-1
	slice_end = min(frame_offset+frame_count, frame_offset+max_frame_idx)

	video = mmcv.VideoReader(path2video)

	for index in range(slice_start, slice_end):
		frame = video[index]

		if (index+1) % 500 == 0:
			print ('-> reading frame %d / %d' % (index+1, slice_end))

		mask = frame_indices == (index - slice_start + 1)
		rows = det_in[mask]

		image_name = "%d" % (index+1)

		bboxes = []
		bbtags = []
		bbimgs = []

		for n,r in enumerate(rows):
			_, x1, y1, x2, y2, s = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])

			# filtering scores less than .51, allowing vals less than .51 causes bug in graph
			if s < 0.51:
				continue
			
			is_reliable = True
			curbox = [x1,y1,x2,y2,s]
			imgbox = frame[int(y1):int(y2), int(x1):int(x2), :]
			# imgbox = extract_patch_block(frame[int(y1):int(y2), int(x1):int(x2), :])

			if not is_patch_reliable(curbox, rows):
				# cv2.imwrite('unreliable_patch_%s_%d.jpg'%(image_name,n), frame[int(y1):int(y2), int(x1):int(x2), :])
				imgbox, is_reliable = find_prev_imgbox(curbox[:4], detections, images, image_name, frame)

			if is_reliable:	
				bbimgs.append( imgbox )
				bboxes.append( curbox )
				bbtags.append( [x1,y1,x2,y2] )
			# else:
			# 	print ('outlier box detected')

		# 3. fill in dictionaries
		detections[image_name] = bboxes
		tags[image_name] = bbtags
		images[image_name] = bbimgs

	print ('-> %d images have been read & processed' % (len(detections)))

	print ('# starting to execute main algorithm')
	
	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	# fib_search = False
	fib_search = True

	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit)
	print ('-> start building min cost flow graph')
	first_img_name = 1 if slice_start == 0 else slice_start
	# tracker.build_network(images, str(first_img_name), str(slice_end-1))
	tracker.build_network(images, str(first_img_name), str(slice_end))
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
	
	source_idx = str(slice_start+1) if slice_start == 0 else str(slice_start) 
	sink_idx = str(slice_end)

	print ('-> offset interval [%s-%s]' % (source_idx, sink_idx))

	for n, (k,_) in enumerate(tracker.flow_dict["source"].items()):
		tr_lst = loop_get_track(k, tracker.flow_dict)
		track_hypot.append(tr_lst)

		s_node = tr_lst[0]
		t_node = tr_lst[-1]

		if s_node[0] != source_idx:
			tr_bgn.append(n)

		if t_node[0] != sink_idx:
			tr_end.append(n)

	print ('# number of unique tracks %d' % (len(track_hypot)))

	print ('-> tracks not finished at sink')
	for index in tr_end:
		print('track index %d finished at frame %s' % (index+1, track_hypot[index][-1]))

	print ('-> tracks not started at source')
	for index in tr_bgn:
		print('track index %d started at frame %s' % (index+1, track_hypot[index][0]))

	# temporal_hungarian_matching(track_hypot, tr_end, tr_bgn, images, detections)

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


	# debug_get_patch_by_id(3, detections, images)
	# debug_print_link_cost(tracker, 3, 13, 3, 865, detections, images)

	return

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

if __name__ == "__main__":
	path2video = sys.argv[1]
	print ('# argv 1 -> path to video file: %s' % (path2video))

	path2det = sys.argv[2]
	print ('# argv 2 -> path to detection file: %s' % (path2det))

	frame_offset = int(sys.argv[3])
	print ('# argv 3 -> frame offset: %d' % (frame_offset))

	frame_count = int(sys.argv[4])
	print ('# argv 4 -> frame count: %d' % (frame_count))

	iid = int(sys.argv[5])
	print ('# argv 5 -> instance id: %d' % (iid))

	visualise = True
	main(path2video, path2det, frame_offset, frame_count, iid)
	
	if visualise == True:
		visualise_hypothesis(path2video, path2det, frame_offset, frame_count)

