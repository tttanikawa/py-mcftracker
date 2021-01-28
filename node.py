class GraphNode:

    def __init__(self, wc, bb, score, is_occl, is_complex, feat_v=None):
        # self._img = patch
        self._3dc = wc
        self._bb = bb
        self._score = score
        # self._tag = tag
        self._is_occl = is_occl
        self._is_complex = is_complex
        self._feat = feat_v
