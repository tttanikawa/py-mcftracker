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

    def onlineTrackerAssign(self, measurements, thresh=5.9915, inf=1e6):
        """
            tracks.predict - new_measurements
            cost_matrix
            linear_assignment
        """

        for track in self.tracks:
            track.predict() # updates track.mean and track.covariance
        
        cost_matrix = np.zeros((len(self.tracks), len(measurements)))

        for i, track in enumerate(self.tracks):
            for j, box in enumerate(measurements):
                dist = track.kf.gating_distance(track.mean, track.covariance, box)
                cost_matrix[i][j] = dist if dist <= thresh else inf

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for row, col in zip(row_indices, col_indices):
            track_idx = self.tracks[row]
            detection_idx = measurements[col]
            matches.append((track_idx, detection_idx))

        return matches

    def onlineTrackerUpdate(measurement):
        """
            tracks.update - new_measurements
        """

        return
