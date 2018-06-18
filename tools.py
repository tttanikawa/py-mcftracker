"""
Copyright 2018 watanika, all rights reserved.
Licensed under the MIT license <LICENSE-MIT or
http://opensource.org/licenses/MIT>. This file may
not be copied, modified,or distributed except
according to those terms.
"""

import cv2


def calc_HS_histogram(image, roi):
	cropped = image[roi[1]:roi[3], roi[0]:roi[2], :]
	hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

	hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
	return hist


def calc_bhattacharyya_distance(hist1, hist2):
	return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def map_node2id(detections):
	node2id = {}
	node2id["source"] = 0
	node2id["sink"] = 1

	nextid = 2
	for image_name, rects in sorted(detections.items()):
		for i, rect in enumerate(rects):
			node2id[(image_name, i, "u")] = nextid
			node2id[(image_name, i, "v")] = nextid + 1
			nextid += 2
	return node2id


def map_id2node(detections):
	id2node = {}
	id2node[0] = "source"
	id2node[1] = "sink"

	nextid = 2
	for image_name, rects in sorted(detections.items()):
		for i, rect in enumerate(rects):
			id2node[nextid] = (image_name, i, "u")
			id2node[nextid + 1] = (image_name, i, "v")
			nextid += 2
	return id2node


def map_name2id(tags):
	name2id = {}
	for frame_id, (image_name, rects) in enumerate(sorted(tags.items())):
		name2id[image_name] = frame_id
	return name2id


def map_id2name(tags):
	id2name = {}
	for frame_id, (image_name, rects) in enumerate(sorted(tags.items())):
		id2name[frame_id] = image_name
	return id2name


def calc_overlap(bb1, bb2):
	bi = (max(bb1[0], bb2[0]), max(bb1[1], bb2[1]), min(bb1[2], bb2[2]), min(bb1[3], bb2[3]))
	iw = bi[2] - bi[0] + 1
	ih = bi[3] - bi[1] + 1
	if iw > 0 and ih > 0:
		ua = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1) + (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1) - iw * ih
		return iw * ih / ua
	else:
		return 0.0
