
import time
from mcftracker import MinCostFlowTracker
import sys
import numpy as np

import mmcv

import helper
import debug
import tracklet_matching

def run_mfct(path2video, path2det, frame_offset, frame_count, iid, match_video_id):
    print ('# starting to read input frames & detection data')

    det_in = np.loadtxt(path2det, delimiter=',')
    frame_indices = det_in[:, 0].astype(np.int)
    # min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    slice_start = 0 if frame_offset == 0 else frame_offset-1
    slice_end = min(frame_offset+frame_count, frame_offset+max_frame_idx)

    data, transform, size, parity, _wc, lf_i = helper.read_input_data(
        path2det, path2video, slice_start, slice_end, det_in, frame_indices, match_video_id)

    # tracks = tracklet_matching.cost_flow_tracklet(data, transform)
    # helper.write_output_data(tracks, path2det, data, len(data)+1, frame_offset, iid, parity)
    # debug.visualise_tracks(path2video, slice_start, slice_end, _wc, transform, size)
    # return

    start = time.time()
    tracker = MinCostFlowTracker(data, 0, 0.1, 0.1)

    print ('-> start building min cost flow graph')
    tracker.build_network(str(len(data)), transform, size)

    print ('-> finish building min cost flow graph')
    optimal_flow, optimal_cost = tracker.run(25, 300, fib=False)
    end = time.time()

    print("Finished: {} sec".format(end - start))
    print("Optimal number of flow: {}".format(optimal_flow))
    print("Optimal cost: {}".format(optimal_cost))

    out_file = './hypothesis_pre.txt'
    out_video = './out_pre.avi'
    track_hypot, nt, nh, nht = helper.build_hypothesis_lst(tracker.flow_dict, "1", lf_i)
    helper.write_output_data(out_file, track_hypot, path2det, data, len(data)+1, frame_offset, iid, parity)
    debug.visualise_tracks(out_file, path2video, slice_start, slice_end, _wc, transform, size, out_video)

    out_file = './hypothesis.txt'
    out_video = './out.avi'
    tracks = tracklet_matching.cost_flow_tracklet(track_hypot, nh, nt, nht, data, transform)
    helper.write_output_data(out_file, tracks, path2det, data, len(data)+1, frame_offset, iid, parity)
    debug.visualise_tracks(out_file, path2video, slice_start, slice_end, _wc, transform, size, out_video)

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

    run_mfct(path2video, path2det, frame_offset, frame_count, iid, mvi)