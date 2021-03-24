"""
Copyright 2018 watanika, all rights reserved.
Licensed under the MIT license <LICENSE-MIT or
http://opensource.org/licenses/MIT>. This file may
not be copied, modified,or distributed except
according to those terms.
"""

import cv2
import sys
sys.path.append("../mmdetection/")
import mmcv
import numpy as np

def get_key(key):
	try:
		return int(key)
	except ValueError:
		return key

def calc_RGB_histogram(image, mask):
	hist = cv2.calcHist([image], [0, 1, 2], mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
	return hist

def calc_HS_histogram(image, mask):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [180, 256], [0, 180, 0, 256])
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
	return hist

def calc_bhattacharyya_distance(hist1, hist2):
	return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def map_node2id(data):
	node2id = {}
	node2id["source"] = 0
	node2id["sink"] = 1

	nextid = 2
	for image_name, node_lst in sorted(data.items(), key=lambda t: get_key(t[0])):
		for i, node in enumerate(node_lst):
			node2id[(image_name, i, "u")] = nextid
			node2id[(image_name, i, "v")] = nextid + 1
			nextid += 2
	return node2id


def map_id2node(data):
	id2node = {}
	id2node[0] = "source"
	id2node[1] = "sink"

	nextid = 2
	for image_name, node_lst in sorted(data.items(), key=lambda t: get_key(t[0])):
		for i, node in enumerate(node_lst):
			id2node[nextid] = (image_name, i, "u")
			id2node[nextid + 1] = (image_name, i, "v")
			nextid += 2
	return id2node

def map_name2id(data):
	name2id = {}
	for frame_id, (image_name, node_lst) in enumerate(sorted(data.items(), key=lambda t: get_key(t[0]))):
		name2id[image_name] = frame_id
	return name2id


def map_id2name(data):
	id2name = {}
	for frame_id, (image_name, node_lst) in enumerate(sorted(data.items(), key=lambda t: get_key(t[0]))):
		id2name[frame_id] = image_name
	return id2name

def calc_overlap(bb1, bb2, dbg=False):
	# iw = bi[2] - bi[0] + 1
	# ih = bi[3] - bi[1] + 1
	
	# if dbg == True:
		# print (iw, ih)
	bi = (max(bb1[0], bb2[0]), max(bb1[1], bb2[1]), min(bb1[2], bb2[2]), min(bb1[3], bb2[3]))
	iw = max(0., bi[2] - bi[0] + 1)
	ih = max(0., bi[3] - bi[1] + 1)
	ua = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1) + (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1) - iw * ih
	return iw * ih / float(ua)
