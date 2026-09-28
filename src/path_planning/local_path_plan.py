from ._path import PathNodeList
from .geometric_map import GeometricMap
from .visibility import VisibilityPathFinder


class LocalPathPlanner:
    """Plans a reference path between two points on a `GeometricMap`'s inflated geometry."""

    def __init__(self, graph_map: GeometricMap, verbose=False):
        self.path_planner = VisibilityPathFinder(graph_map.processed_boundary_coords,
                                                 graph_map.processed_obstacle_list, verbose=verbose)

    def get_ref_path(self, start: tuple, end: tuple) -> PathNodeList:
        # The visibility planner works on (x, y) only; drop any heading on start/end.
        ref_path, dist = self.path_planner.get_ref_path(start[:2], end[:2])
        self.ref_path = PathNodeList.from_tuples(ref_path)
        return self.ref_path
