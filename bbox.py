import numpy as np

class Box(object):
    def __init__(self, tlbr, confidence):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.confidence = float(confidence)

    def to_tlwh(self):
        tlbr = self.tlbr.copy()
        w = tlbr[2]-tlbr[0]
        h = tlbr[3]-tlbr[1]
        x, y = tlbr[0], tlbr[1]
        tlwh = [x,y,w,h]
        return tlwh