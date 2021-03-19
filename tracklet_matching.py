from operator import itemgetter
from TrackletNode import TrackletNode
from mcftracker import MinCostFlowTracker
import helper

import json
import numpy as np
from node import GraphNode

def tracklet_matching(tracklets, nh, nt, nht, data):
    # create data sctruct
    types = [nt, nht, nh]
    tracklet_data = {}

    # sort node_lst by index of last frame
    # for ilst in types:
    #     ends = []
    #     for ti in ilst:            
    #         s, e = tracklets[ti][0], tracklets[ti][-1]
    #         # ends.append(int(e[0]))
    #         ends.append(int(s[0]))
        
    #     ordIdx = [i[0] for i in sorted(enumerate(ends), key=lambda x:x[1])]
    #     indices_s.append(ordIdx)

    #     types_s.append([ilst[i] for i in ordIdx])
        
    # for i, ilst in enumerate(types_s):
    for i, ilst in enumerate(types):
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
        
        tracklet_data[str(i+1)] = node_lst

    return tracklet_data, types

def hypot2id(hypot, ttype, all_tracklets, data_in, transform):
    matches = []
    for hpt in hypot:
        match = []
        for n,h in enumerate(hpt):
            if n%2 != 0:
                sIdx = h[1]
                sTypeIdx = int(h[0])-1
                # oIdx = idx_data[sTypeIdx][sIdx]
                # idM = ttype[sTypeIdx][oIdx]
                idM = ttype[sTypeIdx][sIdx]
                match.append((idM, sTypeIdx+1))
        matches.append(match)

    for mlst in matches:
        # print frame gap and distance between matching pairs
        print (mlst)
        for i in range(len(mlst)-1):
            cur, nex = mlst[i], mlst[i+1]

            curType, curId  = cur[1]-1, cur[0]
            nexType, nexId  = nex[1]-1, nex[0]

            curNode = all_tracklets[curId][-1]
            nexNode = all_tracklets[nexId][0]

            curFname, curNidx = curNode[0], curNode[1]
            nexFname, nexNidx = nexNode[0], nexNode[1]

            curWc = data_in[curFname][curNidx]._3dc
            nexWc = data_in[nexFname][nexNidx]._3dc
            
            cx, cy = curWc[0], curWc[1] 
            rx, ry = nexWc[0], nexWc[1]

            dist = helper.calc_eucl_dist([cx*transform.parameter.get("ground_width"),cy*transform.parameter.get("ground_height")], 
                        [rx*transform.parameter.get("ground_width"),ry*transform.parameter.get("ground_height")])

            fdiff = int(nexFname) - int(curFname)

            print ('%d -> %d dist=%f fn:[%d-%d] fdiff=%d' % (nexId, curId, dist, int(curFname), int(nexFname), fdiff))
        
        print ('-----------------------------------------------------------')
    
    return matches

def interpolate_gap(hypothesis, data, s, e):
    ns = hypothesis[s][-1] # tuple ('frame_num', detection_index, 'u')
    ne = hypothesis[e][0]

    fs = int(ns[0])
    fe = int(ne[0])

    bs = data[ns[0]][ns[1]]._bb
    be = data[ne[0]][ne[1]]._bb

    if fe-fs >= 20:
        return

    # print ('%d -> %d > %s -> %s' % (fs,fe,bs,be))

    fpx1 = [bs[0], be[0]]
    fpy1 = [bs[1], be[1]]
    fpx2 = [bs[2], be[2]]
    fpy2 = [bs[3], be[3]]

    xp = [fs, fe]

    for x in range(fs+1, fe):
        pix1 = np.interp(x, xp, fpx1)
        piy1 = np.interp(x, xp, fpy1)
        pix2 = np.interp(x, xp, fpx2)
        piy2 = np.interp(x, xp, fpy2)

        print ('%d -> %s' % (x, [pix1,piy1,pix2,piy2]))
        pi = [pix1,piy1,pix2,piy2]

        gn = GraphNode([0.,0.], pi, 0., 0)
        fn = str(x)

        index = len(data[fn])
        data[fn].append(gn)

        node_u = (fn, index, "u")
        node_v = (fn, index, "v")

        hypothesis[s].append(node_u)
        hypothesis[s].append(node_v)

    return

def cost_flow_tracklet(assoc_tracklets, nh, nt, nht, data_in, transform):
# def cost_flow_tracklet(data_in, transform):
    # with open('./tracklets.json') as f:
    #     assoc_tracklets = json.load(f)

    # with open('./nh.json') as f:
    #     nh = json.load(f)

    # with open('./nt.json') as f:
    #     nt = json.load(f)

    # with open('./nht.json') as f:
    #     nht = json.load(f)

    # with open('./tracklets.json', 'w') as f:
    #     json.dump(assoc_tracklets, f)

    # with open('./nh.json', 'w') as f:
    #     json.dump(nh, f)

    # with open('./nt.json', 'w') as f:
    #     json.dump(nt, f)

    # with open('./nht.json', 'w') as f:
    #     json.dump(nht, f)

    print ('nh', nh)
    print ('nt', nt)
    print ('nht', nht)

    trklt_data, type_data = tracklet_matching(assoc_tracklets, nh, nt, nht, data_in)
    graph = MinCostFlowTracker(trklt_data, 0, 0.1, 0.1)
    graph.build_network_tracklet(transform)
    # optimal_flow, optimal_cost = graph.run(0, max(len(trklt_data["1"]), len(trklt_data["3"]))+1, fib=False)
    optimal_flow, optimal_cost = graph.run(0, 100, fib=False)
    print("Optimal number of flow: {}".format(optimal_flow))
    print("Optimal cost: {}".format(optimal_cost))
    hypot, _, _, _ = helper.build_hypothesis_lst(graph.flow_dict, "1", "3")
    
    # convert hypot to format: id1 -> id2 -> id3 
    matches = hypot2id(hypot,type_data, assoc_tracklets, data_in, transform)

    # combining tracks	
    for match in matches:
        trklt_s = assoc_tracklets[match[0][0]]
        for i in range(1,len(match)):
            trklt_d = assoc_tracklets[match[i][0]]
            interpolate_gap(assoc_tracklets, data_in, match[0][0], match[i][0])
            for node in trklt_d:
                trklt_s.append(node)
    
    # erasing old tracks
    for match in matches:
        for i in range(1,len(match)):
            assoc_tracklets[match[i][0]].clear()

    return assoc_tracklets

