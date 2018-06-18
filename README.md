# Multi-object tracking using min-cost flow

This is a simple Python implementation of tracking algorithm based on global data association via network flows [1].

Targets are tracked by minimizing network costs built on initial detection results.

## Dependencies

- numpy
- OpenCV (for image reading, processing)
- ortools (for optimizing min-cost flow)

## Usage

Please modify test.py and mcftracker.py to adapt your tracking targets.
You can test this implementation as:

```sh
% python test.py
```

To include it in your project, you just need to:

```py

tracker = MinCostFlowTracker(some_parameters)
tracker.build_network(images)
optimal_flow, optimal_cost = tracker.run()

```

You can use fibonacci search to reduce computation costs.

## License

MIT

## References

[1] L. Zhang et al., "Global data association for multi-object tracking using network flows", CVPR 2008