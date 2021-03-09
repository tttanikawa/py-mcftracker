from operator import itemgetter
from TrackletNode import TrackletNode
from mcftracker import MinCostFlowTracker
import helper

import json

def tracklet_matching(tracklets, nh, nt, nht, data):
    # create data sctruct
    types = [nt, nht, nh]
    types_s = []
    indices_s = []
    tracklet_data = {}

    # sort node_lst by index of last frame
    for ilst in types:
        ends = []
        for ti in ilst:            
            _, e = tracklets[ti][0], tracklets[ti][-1]
            ends.append(int(e[0]))
        
        ordIdx = [i[0] for i in sorted(enumerate(ends), key=lambda x:x[1])]
        indices_s.append(ordIdx)

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
        
        tracklet_data[str(i+1)] = node_lst

    return tracklet_data, indices_s, types

def hypot2id(hypot, idx_data, ttype, all_tracklets, data_in, transform):
    matches = []
    for hpt in hypot:
        match = []
        for n,h in enumerate(hpt):
            if n%2 != 0:
                sIdx = h[1]
                sTypeIdx = int(h[0])-1
                oIdx = idx_data[sTypeIdx][sIdx]
                idM = ttype[sTypeIdx][oIdx]
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

            print ('%d -> %d dist=%f fdiff=%d' % (nexId, curId, dist, fdiff))
        
        print ('-----------------------------------------------------------')

# def cost_flow_tracklet(assoc_tracklets, nh, nt, nht, data_in, transform):
def cost_flow_tracklet(data_in, transform):

    with open('./tracklets.json') as f:
        assoc_tracklets = json.load(f)

    with open('./nh.json') as f:
        nh = json.load(f)

    with open('./nt.json') as f:
        nt = json.load(f)

    with open('./nht.json') as f:
        nht = json.load(f)

    trklt_data, idx_data, type_data = tracklet_matching(assoc_tracklets, nh, nt, nht, data_in)

    # with open('./tracklets.json', 'w', encoding='utf-8') as f:
    #     json.dump(assoc_tracklets, f, ensure_ascii=False)
    
    # with open('./nh.json', 'w', encoding='utf-8') as f:
    #     json.dump(nh, f, ensure_ascii=False)

    # with open('./nt.json', 'w', encoding='utf-8') as f:
    #     json.dump(nt, f, ensure_ascii=False)

    # with open('./nht.json', 'w', encoding='utf-8') as f:
    #     json.dump(nht, f, ensure_ascii=False)

    graph = MinCostFlowTracker(trklt_data, 0, 0.1, 0.1)
    graph.build_network_tracklet(transform)
    optimal_flow, optimal_cost = graph.run(0, max(len(trklt_data["3"]), max(len(trklt_data["1"]), len(trklt_data["2"])))+1, fib=False)
    print("Optimal number of flow: {}".format(optimal_flow))
    print("Optimal cost: {}".format(optimal_cost))
    hypot, _, _, _ = helper.build_hypothesis_lst(graph.flow_dict, "1", "3")
    
    # convert hypot to format: id1 -> id2 -> id3 
    hypot2id(hypot, idx_data, type_data, assoc_tracklets, data_in, transform)
