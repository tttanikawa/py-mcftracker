"""
Copyright 2018 watanika, all rights reserved.
Licensed under the MIT license <LICENSE-MIT or
http://opensource.org/licenses/MIT>. This file may
not be copied, modified,or distributed except
according to those terms.
"""

import math
from ortools.graph import pywrapgraph
import sys
import cv2
import tools
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import helper

class MinCostFlowTracker:
    """
        Object tracking based on data association via minimum cost flow algorithm
        L. Zhang et al.,
        "Global data association for multi-object tracking using network flows",
        CVPR 2008
    """

    def __init__(self, data, min_thresh, P_enter, P_exit):
        self._data = data
        self._min_thresh = min_thresh

        self.P_enter = P_enter
        self.P_exit = P_exit

        self._id2name = tools.map_id2name(data)
        self._name2id = tools.map_name2id(data)

        self._id2node = tools.map_id2node(data)
        self._node2id = tools.map_node2id(data)
        
        self._fib_cache = {0: 0, 1: 1}
        
        self._npast = 30
        self._penalty_skp = 10

    def _fib(self, n):
        if n in self._fib_cache:
            return self._fib_cache[n]
        elif n > 1:
            return self._fib_cache.setdefault(n, self._fib(n - 1) + self._fib(n - 2))
        return n

    def _return_max_dist(self,x):
        if x <= 10:
            return 4.0
        return 4.8*math.log10(x)

    def _find_nearest_fib(self, num):
        for n in range(num):
            if num < self._fib(n):
                return (n - 1, self._fib(n - 1))
        return (num, self._fib(num))

    def _calc_cost_enter(self):
        return -math.log(self.P_enter)

    def _calc_cost_exit(self):
        return -math.log(self.P_exit)

    def _calc_cost_detection(self, beta):
    	return math.log(beta / (1.0 - beta))

    def _calc_cost_link(self, rect1, rect2, image1, image2, dbgLog=False, eps=1e-9):
        prob_iou = tools.calc_overlap(rect1, rect2, dbgLog)
        hist1 = tools.calc_HS_histogram(image1)
        hist2 = tools.calc_HS_histogram(image2)
        prob_color = 1.0 - tools.calc_bhattacharyya_distance(hist1, hist2)

        if prob_color > 0. and prob_iou > 0.:
            prob_sim = prob_iou * prob_color
            return -math.log(prob_sim + eps)
        else:
            return 10000
    
    def _calc_cost_link_appearance(self, prev_node, cur_node, transform, size, dbgLog=False, dst_max=2.4, thresh=16., segment=False):

        if segment: 
            u = prev_node._hist
            v = cur_node._hist
            prob_color = 1.0-tools.calc_bhattacharyya_distance(u, v) 
        else:
            u = prev_node._feat
            v = cur_node._feat            
            prob_color = cosine_similarity([u],[v])[0][0]

        if prev_node._observed:
            u_mean = prev_node._mean
            u_cov = prev_node._covar
            u_kf = prev_node._kf

            sqmah_dist = u_kf.gating_distance(u_mean, u_cov, cur_node._3dc)
            
            if sqmah_dist > thresh:
                return -1 
            
            prob_dst = 1.0 - sqmah_dist / thresh
            prob_sim = 0.7*prob_dst + 0.3*prob_color
            
        else:
            pxy = (prev_node._3dc[0], prev_node._3dc[1])
            cxy = (cur_node._3dc[0], cur_node._3dc[1])

            a = np.array(pxy)
            b = np.array(cxy)

            dst_eucl = np.linalg.norm(a-b)
            prob_dst = np.float32(1.0 - dst_eucl / dst_max)

            if dst_eucl >= dst_max:
                return -1

            bu = prev_node._bb
            bv = cur_node._bb
            prob_iou = tools.calc_overlap(bu,bv)

            if prob_iou != 0.:
                c1, c2, c3 = 0.20, 0.60, 0.20
                prob_sim = c1*prob_dst + c2*prob_color + c3*prob_iou
            else:
                c1, c2, c3 = 0.2, 0.8, 0.0
                prob_sim = c1*prob_dst + c2*prob_color + c3*prob_iou

        return -math.log(prob_sim)

    def build_network(self, last_img_name, transform, size, f2i_factor=100000):
        self.mcf = pywrapgraph.SimpleMinCostFlow()

        for n, (image_name, node_lst) in enumerate(sorted(self._data.items(), key=lambda t: tools.get_key(t[0]))):
            if n % 100 == 0:
                print ('-> processing image %s / %s' % (image_name, last_img_name))

            frame_id = self._name2id[image_name]

            pnlty_en = 1 if frame_id == 0 else 100
            
            for i, node in enumerate(node_lst):
                self.mcf.AddArcWithCapacityAndUnitCost(self._node2id["source"], self._node2id[(image_name, i, "u")], 1, int(self._calc_cost_enter() * f2i_factor * pnlty_en))
                self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "u")], self._node2id[(image_name, i, "v")], 1, int(self._calc_cost_detection(1.0-node._score)*f2i_factor))
                self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "v")], self._node2id["sink"], 1, int(self._calc_cost_exit() * f2i_factor * pnlty_en))
            
            if frame_id > 0:
                prev_image_name = self._id2name[frame_id - 1]
                
                for i, i_node in enumerate(self._data[prev_image_name]):
                    for j, j_node in enumerate(node_lst):
                        unit_cost = self._calc_cost_link_appearance(i_node, j_node, transform, size)
                        if unit_cost >= 0.:
                            self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(prev_image_name, i, "v")], self._node2id[(image_name, j, "u")], 1, int(unit_cost*1000))

                # connect N previous frames to current frame's nodes
                if frame_id >= 2:
                    npast = self._npast if frame_id >= self._npast else frame_id

                    for step in range(npast, 1, -1):
                        fpast = self._id2name[frame_id-step]
                        for i, i_node in enumerate(self._data[fpast]):
                            for j, j_node in enumerate(node_lst):
                                unit_cost = self._calc_cost_link_appearance(i_node, j_node, transform, size)
                                if unit_cost >= 0.:
                                    self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(fpast, i, "v")], self._node2id[(image_name, j, "u")], 1, int(unit_cost*1000*self._penalty_skp))

        return

    def _make_flow_dict(self):
        self.flow_dict = {}
        for i in range(self.mcf.NumArcs()):
            if self.mcf.Flow(i) > 0:
                tail = self.mcf.Tail(i)
                head = self.mcf.Head(i)
                if self._id2node[tail] in self.flow_dict:
                    self.flow_dict[self._id2node[tail]][self._id2node[head]] = 1
                else:
                    self.flow_dict[self._id2node[tail]] = {self._id2node[head]: 1}

    def _fibonacci_search(self, search_range=200):
        s = 0
        k_max, t = self._find_nearest_fib(self.mcf.NumNodes() // search_range)

        print ('number of nodes: %d | maximum flows: %d' % (self.mcf.NumNodes(), k_max))
        cost = {}

        for k in range(k_max, 1, -1):
            # s < u < v < t
            u = s + self._fib(k - 2)
            v = s + self._fib(k - 1)

            if u not in cost:
                self.mcf.SetNodeSupply(self._node2id["source"], u)
                self.mcf.SetNodeSupply(self._node2id["sink"], -u)

                if self.mcf.Solve() == self.mcf.OPTIMAL:
                    cost[u] = self.mcf.OptimalCost()
                else:
                    print("There was an issue with the min cost flow input.")
                    sys.exit()

            if v not in cost:
                self.mcf.SetNodeSupply(self._node2id["source"], v)
                self.mcf.SetNodeSupply(self._node2id["sink"], -v)

                if self.mcf.Solve() == self.mcf.OPTIMAL:
                    cost[v] = self.mcf.OptimalCost()
                else:
                    print("There was an issue with the min cost flow input.")
                    sys.exit()

            if cost[u] < cost[v]:
                t = v
            elif cost[u] == cost[v]:
                s = u
                t = v
            else:
                s = u

        self.mcf.SetNodeSupply(self._node2id["source"], s)
        self.mcf.SetNodeSupply(self._node2id["sink"], -s)

        if self.mcf.Solve() == self.mcf.OPTIMAL:
            optimal_cost = self.mcf.OptimalCost()
        else:
            print("There was an issue with the min cost flow input.")
            sys.exit()
        self._make_flow_dict()
        return (s, optimal_cost)

    def _brute_force(self, min_flow, max_flow, search_range=100):
        optimal_flow = -1
        optimal_cost = float("inf")

        print ('min flow: %d max flow: %d' % (min_flow, max_flow))

        for flow in range(min_flow, max_flow):
            self.mcf.SetNodeSupply(self._node2id["source"], flow)
            self.mcf.SetNodeSupply(self._node2id["sink"], -flow)

            if self.mcf.Solve() == self.mcf.OPTIMAL:
                cost = self.mcf.OptimalCost()
            else:
                continue

            if cost < optimal_cost:
                optimal_flow = flow
                optimal_cost = cost
                self._make_flow_dict()

            # print ('flow: %d optimal flow: %d' % (flow, optimal_flow))
        
        return (optimal_flow, optimal_cost)

    def run(self, min_flow, max_flow, fib=False):
        if fib:
            return self._fibonacci_search()
        else:
            return self._brute_force(min_flow, max_flow)

