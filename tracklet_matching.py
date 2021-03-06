from operator import itemgetter
from TrackletNode import TrackletNode
from mcftracker import MinCostFlowTracker

def tracklet_matching(tracklets, nh, nt, nht, data):
    # tintrv = []
    # for t_i in nht:
    #     s, e = tracklets[t_i][0], tracklets[t_i][-1]
    #     tintrv.append((t_i, int(s[0]), int(e[0])))

    # # sort nht by end frame
    # tintrv_s = sorted(tintrv,key=itemgetter(2))

    # groups, indices = [], []
    
    # # find overlaps
    # for i, i1 in enumerate(tintrv_s):
    #     if i not in indices:
    #         cs, ce = i1[1], i1[2]

    #         group = [i1]
    #         indices.append(i)

    #         for j, i2 in enumerate(tintrv_s):
    #             if j not in indices:
    #                 rs, re = i2[1], i2[2]
    #                 if rs <= ce:
    #                     group.append(i2)
    #                     indices.append(j)

    #         groups.append(group)

    # for g in groups:
    #     print (g)

    # create data sctruct
    types = [nt, nht, nh]
    types_s = []
    tracklet_data = {}

    # sort node_lst by index of last frame
    for ilst in types:
        ends = []
        for ti in ilst:            
            _, e = tracklets[ti][0], tracklets[ti][-1]
            ends.append(int(e[0]))
        
        ordIdx = [i[0] for i in sorted(enumerate(ends), key=lambda x:x[1])]
        types_s.append([ilst[i] for i in ordIdx])
        
    for i, ilst in enumerate(types_s):
        node_lst = []

        for idx in ilst:
            tf = tracklets[idx][0]
            _sWc = data[tf[0]][tf[1]]._3dc
            _sFn = int(tf[0])

            tl = tracklets[idx][-1]
            _eWc = data[tl[0]][tl[1]]._3dc
            _eFn = int(tl[0])

            # _wcS coord of first observation of a tracklet
            # _wcE coord of last observation of a tracklet
            # _sFn start frame index of a tracklet
            # _eFn end frame index of a tracklet

            node = TrackletNode(idx, _sFn, _sWc, _eFn, _eWc)
            node_lst.append(node)
        
        tracklet_data[str(i)] = node_lst

    return tracklet_data

def cost_flow_tracklet(data):

    graph = MinCostFlowTracker(data, 0, 0.2, 0.1)
    # graph.build_network_tracklet()
