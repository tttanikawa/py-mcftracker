
import time
from mcftracker import MinCostFlowTracker
import sys
import numpy as np

import mmcv

import helper
import debug


def run_mfct(path2video, path2det, frame_offset, frame_count, iid, match_video_id):
    print ('# starting to read input frames & detection data')

    det_in = np.loadtxt(path2det, delimiter=',')
    frame_indices = det_in[:, 0].astype(np.int)
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    slice_start = 0 if frame_offset == 0 else frame_offset-1
    slice_end = min(frame_offset+frame_count, frame_offset+max_frame_idx)

    detections, tags, images, features = helper.read_input_data(
        path2det, path2video, slice_start, slice_end, det_in, frame_indices)

    start = time.time()
    tracker = MinCostFlowTracker(detections, tags, 0, 0.1, 0.1)

    print ('-> start building min cost flow graph')
    first_img_name = 1 if slice_start == 0 else slice_start
    tracker.build_network(images, features, str(first_img_name), str(slice_end))

    print ('-> finish building min cost flow graph')
    optimal_flow, optimal_cost = tracker.run(fib=True)
    end = time.time()

    print("Finished: {} sec".format(end - start))
    print("Optimal number of flow: {}".format(optimal_flow))
    print("Optimal cost: {}".format(optimal_cost))

    tr_end = []
    tr_bgn = []

    track_hypot = []

    source_idx = str(slice_start+1) if slice_start == 0 else str(slice_start)
    sink_idx = str(slice_end)

    print ('-> offset interval [%s-%s]' % (source_idx, sink_idx))

    for n, (k, _) in enumerate(tracker.flow_dict["source"].items()):
        tr_lst = helper.loop_get_track(k, tracker.flow_dict)
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
        print('track index %d finished at frame %s' %
              (index+1, track_hypot[index][-1]))

    print ('-> tracks not started at source')
    for index in tr_bgn:
        print('track index %d started at frame %s' %
              (index+1, track_hypot[index][0]))

    # helper.temporal_hungarian_matching(track_hypot, tr_end, tr_bgn, images, detections)
    helper.temporal_hungarian_matching(track_hypot, tr_end, tr_bgn, features, detections, match_video_id)
    
    helper.write_output_data(track_hypot, path2det, detections, slice_start, slice_end, frame_offset, iid)
    # debug.get_patch_by_id(tracker, 17, detections, images, features)

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

    mvi = int(sys.argv[6])
    print ('# argv 6 -> match_video_id: %d' % (mvi))

    visualise = True
    run_mfct(path2video, path2det, frame_offset, frame_count, iid, mvi)

    if visualise == True:
        debug.visualise_hypothesis(
            path2video, path2det, frame_offset, frame_count)
