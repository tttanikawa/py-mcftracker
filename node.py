class GraphNode:

    def __init__(self, wc, bb, score, status, feat_v=None):
        self._3dc = wc
        self._bb = bb
        self._score = score
        # self._is_occl = is_occl
        # self._is_complex = is_complex
        self._status = status
        self._feat = feat_v
