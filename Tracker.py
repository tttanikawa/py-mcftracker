from Track import OnlineTrack
import numpy as np
import KalmanFilter

from scipy.optimize import linear_sum_assignment

class OnlineTracker:

    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def onlineTrackerInit(self, boxes_init):

        for box in boxes_init:
            new_kf = KalmanFilter.KalmanFilter()
            mean, cov = new_kf.initiate(np.asarray(box.to_world()))
            self.tracks.append(OnlineTrack(self._next_id, mean, cov, new_kf))
            self._next_id += 1

        print ('%d tracks created' % (len(self.tracks)))

        return

    def onlineTrackerAssign(self, measurements, fnum, thresh=5.9915, inf=1000000):

        for track in self.tracks:
            track.predict() # updates track.mean and track.covariance
        
        cost_matrix = np.zeros((len(self.tracks), len(measurements)))

        for i, track in enumerate(self.tracks):
            for j, box in enumerate(measurements):
                sq_mah_dist = track.kf.gating_distance(track.mean, track.covariance, box.to_world())
                cost_matrix[i][j] = inf if sq_mah_dist >= thresh else sq_mah_dist

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row][col] == inf:
                # print ('f:%d id:%d cost:%f' % (fnum, self.tracks[row].id, cost_matrix[row][col]))
                unmatched_tracks.append(row)
                unmatched_dets.append(col)
            else:
                matches.append((row, col))

        for i, _ in enumerate(self.tracks):
            if i not in row_indices:
                unmatched_tracks.append(i)

        for j, _ in enumerate(measurements):
            if j not in col_indices:
                unmatched_dets.append(j)

        for tidx in unmatched_tracks:
            print ('f:%d id:%d unmatched' % (fnum, self.tracks[tidx].id))

        return matches, unmatched_tracks, unmatched_dets

    def onlineTrackerUpdate(self, matches, measurements):
        """
            tracks.update - new_measurements
        """
        
        for match in matches:
            self.tracks[match[0]].update(measurements[match[1]].to_world())
