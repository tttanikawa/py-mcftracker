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

# for cosine difference
from scipy.spatial import distance
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
        self.P_exit = self.P_enter

        self._id2name = tools.map_id2name(data)
        self._name2id = tools.map_name2id(data)

        self._id2node = tools.map_id2node(data)
        self._node2id = tools.map_node2id(data)
        
        self._fib_cache = {0: 0, 1: 1}

    def _fib(self, n):
        if n in self._fib_cache:
            return self._fib_cache[n]
        elif n > 1:
            return self._fib_cache.setdefault(n, self._fib(n - 1) + self._fib(n - 2))
        return n

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

    def _calc_cost_link_appearance(self, prev_node, cur_node, transform, size, dbgLog=False, eps=1e-7):
        alpha  = 0.5
        cos_max = 0.300
        dis_max = 1.8

        if prev_node._status == 1:
            alpha  = 0.3
            cos_max = 0.1600
        elif prev_node._status == 2:
            alpha  = 0.8
        elif prev_node._status == 3 or prev_node._status == 4:
            return 10000

        u = prev_node._feat
        v = cur_node._feat

        rect1 = prev_node._bb
        rect2 = cur_node._bb

        cos_dist = distance.cosine(u, v)
        
        if cos_dist > cos_max:
            return 10000

        p1 = helper.box2midpoint_normalised(rect1, size[1], size[0])
        p2 = helper.box2midpoint_normalised(rect2, size[1], size[0])

        cx, cy = transform.video_to_ground(p1[0], p1[1])
        rx, ry = transform.video_to_ground(p2[0], p2[1])

        dist = helper.calc_eucl_dist([cx*transform.parameter.get("ground_width"),cy*transform.parameter.get("ground_height")], 
                            [rx*transform.parameter.get("ground_width"),ry*transform.parameter.get("ground_height")])

        if dist >= dis_max:
            return 10000

        dist_norm = dist / dis_max
        prob_dist = 1.0 - dist_norm
        prob_color = 1.0 - cos_dist

        prob_sim = alpha*prob_dist + (1.0-alpha)*prob_color

        return -math.log(prob_sim)

    def build_network(self, first_img_name, last_img_name, transform, size, f2i_factor=200):
        self.mcf = pywrapgraph.SimpleMinCostFlow()

        for n, (image_name, node_lst) in enumerate(sorted(self._data.items(), key=lambda t: tools.get_key(t[0]))):

            if n % 100 == 0:
                print ('-> processing image %s / %s' % (image_name, last_img_name))

            f2i_en = 10000
            f2i_ex = 10000

            for i, node in enumerate(node_lst):
                if node._status != 3 and node._status != 4:
                    self.mcf.AddArcWithCapacityAndUnitCost(self._node2id["source"], self._node2id[(image_name, i, "u")], 1, int(self._calc_cost_enter() * f2i_en))
                    self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "u")], self._node2id[(image_name, i, "v")], 1, int(self._calc_cost_detection(1.0-node._score) * f2i_factor))
                    self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "v")], self._node2id["sink"], 1, int(self._calc_cost_exit() * f2i_ex))

            frame_id = self._name2id[image_name]
            
            if frame_id == 0:
                continue

            prev_image_name = self._id2name[frame_id - 1]
            if prev_image_name not in self._data:
                continue
            
            for i, i_node in enumerate(self._data[prev_image_name]):
                for j, j_node in enumerate(node_lst):
                    unit_cost = int(self._calc_cost_link_appearance(i_node, j_node, transform, size) * 100)
                    self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(prev_image_name, i, "v")], self._node2id[(image_name, j, "u")], 1, unit_cost)
                
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

    def _brute_force(self, search_range=100):
        optimal_flow = -1
        optimal_cost = float("inf")

        for flow in range(50,80):
            self.mcf.SetNodeSupply(self._node2id["source"], flow)
            self.mcf.SetNodeSupply(self._node2id["sink"], -flow)

            if self.mcf.Solve() == self.mcf.OPTIMAL:
                cost = self.mcf.OptimalCost()
            else:
                print("There was an issue with the min cost flow input.")
                sys.exit()

            print ("amount of flow at source: %d / optimal cost: %d" % (flow, cost))

            if cost < optimal_cost:
                optimal_flow = flow
                optimal_cost = cost
                self._make_flow_dict()
                
        return (optimal_flow, optimal_cost)

    def run(self, fib=False):
        if fib:
            return self._fibonacci_search()
        else:
            return self._brute_force()
