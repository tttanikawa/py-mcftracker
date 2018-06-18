"""
Copyright 2018 watanika, all rights reserved.
Licensed under the MIT license <LICENSE-MIT or
http://opensource.org/licenses/MIT>. This file may
not be copied, modified,or distributed except
according to those terms.
"""

import time
from mcftracker import MinCostFlowTracker


# Example usage of mcftracker
def main():
	# Prepare initial detecton results, ground truth, and images
	# You need to change below
	detections = {"image_name": [x1, y1, x2, y2, score]}
	tags = {"image_name": [x1, y1, x2, y2]}
	images = {"image_name": numpy_image}

	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	beta = 0.5
	fib_search = True

	# Let's track them!
	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit, beta)
	tracker.build_network(images)
	optimal_flow, optimal_cost = tracker.run(fib=fib_search)
	end = time.time()

	print("Finished: {} sec".format(end - start))
	print("Optimal number of flow: {}".format(optimal_flow))
	print("Optimal cost: {}".format(optimal_cost))

	print("Optimal flow:")
	print(tracker.flow_dict)


if __name__ == "__main__":
	main()
