"""Shortest paths through free space on a visibility graph of (inflated) obstacle vertices."""
import networkx as nx # type: ignore
from shapely.geometry import Point, Polygon, LineString # type: ignore


PathNode = tuple[float, float]


class VisibilityPathFinder:
    def __init__(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]], verbose=False) -> None:
        self._boundary_polygon = Polygon(boundary_coords)
        self._obstacle_polygons = [Polygon(obs) for obs in obstacle_list]
        self._all_vertices = [x for y in obstacle_list for x in y]
        self.vb = verbose

        self._pre_G = self._pre_build()

    def _is_visible(self, p1: PathNode, p2: PathNode) -> bool:
        line = LineString([p1, p2])
        # The whole segment must stay inside the boundary, not just its endpoints: an inflated
        # obstacle corner can land outside the map, and a path through it would leave the map.
        if not self._boundary_polygon.contains(line):
            return False
        return not any(poly.intersects(line) and not poly.touches(line) for poly in self._obstacle_polygons)

    def _pre_build(self):
        G = nx.Graph()
        for i, point1 in enumerate(self._all_vertices):
            for j, point2 in enumerate(self._all_vertices[i+1:], i+1):
                if self._is_visible(point1, point2):
                    line = LineString([point1, point2])
                    G.add_edge(i+2, j+2, weight=line.length) # 0, 1 are start and end points
        return G

    def get_ref_path(self, start: PathNode, end: PathNode) -> tuple[list[PathNode], list[float]]:
        """Get the reference path from start to end.

        Args:
            start/end: The start/end point of the path. PathNode = tuple[float, float].

        Raises:
            ValueError: If the start/end point is in an obstacle or outside the boundary.

        Returns:
            shortest_path_coords: List of coordinates/PathNode of the shortest path.
            section_lengths: List of lengths of the sections of the shortest path.
        """
        if any(Point(start).intersects(poly) for poly in self._obstacle_polygons):
            raise ValueError('Start point is in an obstacle.')
        if not self._boundary_polygon.contains(Point(start)):
            raise ValueError('Start point is outside the boundary.')
        if any(Point(end).intersects(poly) for poly in self._obstacle_polygons):
            raise ValueError('End point is in an obstacle.')
        if not self._boundary_polygon.contains(Point(end)):
            raise ValueError('End point is outside the boundary.')
        
        points = [start, end] + self._all_vertices
        G = self._pre_G.copy()
        for i, point1 in enumerate(points[:2]):
            for j, point2 in enumerate(points[i+1:], i+1):
                if self._is_visible(point1, point2):
                    line = LineString([point1, point2])
                    G.add_edge(i, j, weight=line.length)

        shortest_path = nx.shortest_path(G, source=0, target=1, weight='weight')
        shortest_path_coords = [points[i] for i in shortest_path]
        section_lengths: list[float] = [G.edges[shortest_path[i], shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1)]
        return shortest_path_coords, section_lengths

