from torchreid.utils import FeatureExtractor
import torch

import os
import sys
import numpy as np
import cv2

from PIL import Image
import errno

import mmcv 

def test_sanity():
    features = extractor('/home/dmitriy.khvan/deep-person-reid/tmp/debug/dbg_orig_0.jpg')
    print(features)

def extract_image_patch(image, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    patch = image[y:y+h, x:x+w]
    return patch

def network_feed(image, boxes, extractor):
    patches_np_lst = []

    for num, box in enumerate(boxes):
        patch = extract_image_patch(image, box)
        # cv2.imwrite('./tmp/debug/dbg_orig_%d.jpg' % (num), patch)
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if patch_rgb is None:
            print("WARNING: Failed to extract image patch: %s." % str(box))
            patch_rgb = np.random.uniform(0., 255., image_shape).astype(np.uint8)

        patch_np = np.array(patch_rgb)
        patches_np_lst.append(patch_np)

    features = extractor(patches_np_lst)
    features_np = features.cpu().detach().numpy()
    
    return features_np

def network_feed_from_list(image_list, extractor):
    patches_np_lst = []

    for patch in image_list:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if patch_rgb is None:
            print("WARNING: Failed to extract image patch: %s." % str(box))
            patch_rgb = np.random.uniform(0., 255., image_shape).astype(np.uint8)

        patch_np = np.array(patch_rgb)
        patches_np_lst.append(patch_np)

    features = extractor(patches_np_lst)
    features_np = features.cpu().detach().numpy()

    return features_np

def generate_detections(ckpt_path, mot_dir, output_dir, detection_dir=None):

    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=ckpt_path,
        device='cuda'
    )

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(mot_dir, "img1")

        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(mot_dir, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        # output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        output_filename = os.path.join(output_dir, "features.npy")

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
        
            bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)

            features = network_feed(bgr_image, rows[:, 2:6].copy(), extractor)
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)
        break

def generate_detections_from_video(frame_offset, frame_count, ckpt_path, video_path, det_file_path, output_dir):

    try:
        os.makedirs(output_dir)
    
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=ckpt_path,
        device='cuda'
    )

    frame_offset = int(frame_offset)
    frame_count = int(frame_count)

    video = mmcv.VideoReader(video_path)

    slice_start = 0 if frame_offset == 0 else frame_offset-1
    slice_end = frame_offset+frame_count

    output_filename = os.path.join(output_dir, "features.npy")
    detections_in = np.loadtxt(det_file_path, delimiter=',')
    frame_indices = detections_in[:, 0].astype(np.int)

    print('[DSORT] processing frames from {} - {}'.format(range(slice_start,slice_end)[0], range(slice_start,slice_end)[-1]))

    detections_out = []

    for index in range(slice_start, slice_end):
        print("Frame %05d/%05d" % (index, slice_end))

        mask = frame_indices == (index - slice_start + 1)
        rows = detections_in[mask]

        frame = video[index]
        
        if frame is None:
            print('[DBG] Empty frame received!')
            break

        features = network_feed(frame, rows[:, 2:6].copy(), extractor)
        detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

    np.save(output_filename, np.asarray(detections_out), allow_pickle=False)

if __name__=="__main__": 
    # sys.argv[1] - starting frame index in video
    # sys.argv[2] - number of frames within interval
    # sys.argv[3] - path to feature extractor checkpoint
    # sys.argv[4] - path to video file
    # sys.argv[5] - path to detection txt file
    # sys.argv[6] - path to output directory

    generate_detections_from_video(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

