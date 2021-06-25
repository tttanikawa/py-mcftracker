import numpy as np

class Box(object):
    def __init__(self, tlbr, confidence, transform, imsize):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.confidence = float(confidence)
        self.transform = transform
        self.size = imsize

    def to_tlwh(self):
        tlbr = self.tlbr.copy()
        w = tlbr[2]-tlbr[0]
        h = tlbr[3]-tlbr[1]
        x, y = tlbr[0], tlbr[1]
        tlwh = [x,y,w,h]
        return tlwh

    def box2midpoint_normalised(self, box, iw, ih):
        w = box[2]-box[0]
        x, y = box[0] + w/2, box[3]
        return (x/iw, y/ih)

    def to_world(self):
        p = self.box2midpoint_normalised(self.tlbr, self.size[1], self.size[0])
        cx, cy = self.transform.video_to_ground(p[0], p[1])
        cx, cy = cx*self.transform.parameter.get("ground_width"), cy*self.transform.parameter.get("ground_height")
        return np.asarray([cx,cy], dtype=np.float)



