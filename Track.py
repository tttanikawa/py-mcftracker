class OnlineTrack:
    def __init__(self, id, mean, covariance, kf):
        """
            each Track object has mean [x,y,dx,dy] and covariance matrix
            and receives Kalman Filter object
        """
        self.id = id
        self.mean = mean
        self.covariance = covariance
        self.kf = kf

    def update(self, detection):
        """
            updates corresponding Kalman Filter object's parameters & mean and covariance of the track
            detection - is a box in x,y,w,h format
        """
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection)

    def predict(self):
        """
            updates track's mean and covariance
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
