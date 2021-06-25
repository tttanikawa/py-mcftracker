class TrackletNode:
    def __init__(self, idx, sfIdx, s3dc, efIdx, e3dc):
        self._id = idx + 1
        self._sfIdx = sfIdx
        self._efIdx = efIdx
        self._s3dc = s3dc
        self._e3dc = e3dc
