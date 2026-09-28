class GeometricMap:
    """A map's boundary and static obstacles as vertex lists, raw and inflated.

    `processed_*` are the inflated versions (obstacles grown, boundary shrunk by the controller's
    safety margins); they are what the MPC's static constraints and the replanner are built from.
    """
    def __init__(self, boundary_coords: list[tuple], obstacle_list: list[list[tuple]],
                 processed_boundary_coords: list[tuple] | None = None,
                 processed_obstacle_list: list[list[tuple]] | None = None):
        if not isinstance(boundary_coords, list) or not isinstance(obstacle_list, list):
            raise TypeError("A map boundary must be a list of tuples, and its obstacles a list of such lists.")
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.processed_boundary_coords = processed_boundary_coords
        self.processed_obstacle_list = processed_obstacle_list
