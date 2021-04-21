class GraphNode:
    def __init__(self, wc, bb, score, status, feat_v=None, mask=None, hist=None):
        self._3dc = wc
        self._bb = bb
        self._score = score
        self._status = status
        self._feat = feat_v
        self._mask = mask
        self._hist = hist
        
        self._observed = False
        self._mean = None
        self._covar = None
        self._kf = None
