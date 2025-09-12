import sim
import nodify
"""
Cost Estimator for DARP using OR-Tools.

Goal
----
Given:
  - distance_matrix: ndarray of size (L x L) with entries in kilometers
  - depot_node: integer index into the matrix (default 0)
  - requests: list of dicts: {'origin': int, 'destination': int, 'o_t_index': int, 'd_t_index': int}
      * time indices are half-hours: 1 -> [00:00, 00:30), 12 -> [06:00, 06:30)
      * BUT in our model, requests are considered to arrive between 05:00 and 20:00,
        i.e., valid index range [10, 40] inclusive. We clamp to this window.
  - vehicle_num: starting number of vehicles to try
  - vehicle_travel_speed: km/h (default 20.0)
  - vehicle_penalty: penalty added each time we increase vehicles by 1
  - time_window_duration: minutes per index bucket (default 30)
  - vehicle_capacity: int (default 4)
  - max_vehicles: hard cap (default vehicle_num + 10 if not specified)
  - max_solve_time: maximum wall-clock time in seconds for OR-Tools solver (float supported)

Behavior
--------
- We attempt to solve a pickup-and-delivery routing problem (DARP) with given vehicles.
- **Requests are mandatory.** We do NOT add disjunctions (no dropping).
- If infeasible, we increase the number of vehicles by 1 and add `vehicle_penalty`
  to the total cost model. Repeat until feasible or hitting `max_vehicles`.
- The reported cost includes:
    * routing_cost (sum of traveled distance in km),
    * vehicle_penalty_cost ((num_vehicles_used - vehicle_num) * vehicle_penalty),
    * total_cost = routing_cost + vehicle_penalty_cost.

Returns
-------
  Dict with:
    - total_cost
    - routing_cost
    - vehicle_penalty_cost
    - num_vehicles_used
    - num_vehicles_attempted
    - status: 'FEASIBLE' or 'INFEASIBLE'
    - solve_time: seconds
    - routes: list of routes as node ids (excluding depot), for diagnostics
"""

from typing import Dict, List
import time
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ----------------------------- Public API ----------------------------- #

def cost_estimator(
    distance_matrix: np.ndarray,
    requests: List[Dict],
    vehicle_num: int,
    depot_node: int = 0,
    vehicle_travel_speed: float = 20.0,
    vehicle_penalty: float = 1000.0,
    time_window_duration: int = 30,
    vehicle_capacity: int = 4,
    max_vehicles: int = None,
    max_solve_time: float = 10.0,
) -> Dict:
    """
    Estimate the total routing cost for a mandatory-serve DARP using OR-Tools.
    If infeasible with the current number of vehicles, add vehicles one-by-one,
    incurring `vehicle_penalty` each time, until a feasible solution is found
    or `max_vehicles` is reached.
    """
    if max_vehicles is None:
        max_vehicles = max(vehicle_num, 1) + 10

    start_wall = time.time()
    best = None

    for k in range(vehicle_num, max_vehicles + 1):
        sol = _solve_darp_mandatory(
            distance_matrix=distance_matrix,
            depot_node=depot_node,
            requests=requests,
            num_vehicles=k,
            vehicle_travel_speed=vehicle_travel_speed,
            time_window_duration=time_window_duration,
            vehicle_capacity=vehicle_capacity,
            max_solve_time=max_solve_time,
        )

        # Compose cost model for this attempt
        routing_cost_km = sol["total_distance_km"]
        vehicle_penalty_cost = (k - vehicle_num) * vehicle_penalty
        total_cost = routing_cost_km + vehicle_penalty_cost

        # Augment solution dict with cost components
        sol.update(
            {
                "routing_cost": routing_cost_km,
                "vehicle_penalty_cost": vehicle_penalty_cost,
                "total_cost": total_cost,
                "num_vehicles_attempted": k,
            }
        )

        # If feasible, pick the first feasible (lowest k)
        if sol["status"] == "FEASIBLE":
            best = sol
            break

    if best is None:
        # Return the last infeasible attempt's metadata
        best = {
            "status": "INFEASIBLE",
            "routing_cost": 0.0,
            "vehicle_penalty_cost": (max_vehicles - vehicle_num + 1) * vehicle_penalty,  # not used for total
            "total_cost": float("inf"),
            "num_vehicles_used": 0,
            "num_vehicles_attempted": max_vehicles,
            "solve_time": time.time() - start_wall,
            "routes": [],
        }
    else:
        best["solve_time"] = time.time() - start_wall

    return best


# -------------------------- Internal Solver -------------------------- #

def _solve_darp_mandatory(
    distance_matrix: np.ndarray,
    depot_node: int,
    requests: List[Dict],
    num_vehicles: int,
    vehicle_travel_speed: float,
    time_window_duration: int,
    vehicle_capacity: int,
    max_solve_time: float,
) -> Dict:
    """Solve a DARP with mandatory pickup-and-delivery (no dropping)."""

    # Build node sets
    pickup_nodes, dropoff_nodes, node_to_request, request_to_pickup, request_to_dropoff = _create_nodes(requests)
    num_nodes = 1 + len(pickup_nodes) + len(dropoff_nodes)  # depot + pickups + dropoffs
    depot = 0

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback (return integer cost). Keep units consistent: use meters as integer arc costs.
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == to_node:
            return 0
        i = _get_node_location(from_node, depot_node, pickup_nodes, dropoff_nodes, node_to_request, requests)
        j = _get_node_location(to_node, depot_node, pickup_nodes, dropoff_nodes, node_to_request, requests)
        if i is None or j is None:
            return 0
        dist_km = float(distance_matrix[i][j])
        return int(round(dist_km * 1000.0))  # meters

    transit_cost = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cost)

    # Time callback (minutes) using travel speed
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == to_node:
            return 0
        i = _get_node_location(from_node, depot_node, pickup_nodes, dropoff_nodes, node_to_request, requests)
        j = _get_node_location(to_node, depot_node, pickup_nodes, dropoff_nodes, node_to_request, requests)
        if i is None or j is None:
            return 0
        dist_km = float(distance_matrix[i][j])
        # minutes = (km / (km/h)) * 60
        minutes = int(round((dist_km / max(vehicle_travel_speed, 1e-6)) * 60.0))
        return max(0, minutes)

    transit_time = routing.RegisterTransitCallback(time_callback)

    # Time dimension
    # Allow waiting time; set a daily horizon (24h)
    routing.AddDimension(
        transit_time,
        120,    # waiting allowed (buffer)
        1440,   # horizon in minutes
        False,  # do not force start cumul to zero; vehicles can start later
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Allow depot at any time in the day
    for v in range(num_vehicles):
        time_dimension.CumulVar(routing.Start(v)).SetRange(0, 1440)
        time_dimension.CumulVar(routing.End(v)).SetRange(0, 1440)

    # Capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        if node == depot:
            return 0
        return 1 if node in pickup_nodes else (-1 if node in dropoff_nodes else 0)

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity",
    )

    # Time windows (clamped to 05:00–20:00 i.e., indices 10..40)
    def clamp_index(idx: int) -> int:
        return max(10, min(40, int(idx)))

    for i, req in enumerate(requests):
        p_node = request_to_pickup[i]
        d_node = request_to_dropoff[i]

        p_idx = clamp_index(req["o_t_index"])
        d_idx = clamp_index(req["d_t_index"])

        p_start = p_idx * time_window_duration
        p_end = p_start + time_window_duration
        d_start = d_idx * time_window_duration
        d_end = d_start + time_window_duration

        time_dimension.CumulVar(manager.NodeToIndex(p_node)).SetRange(p_start, p_end)
        time_dimension.CumulVar(manager.NodeToIndex(d_node)).SetRange(d_start, d_end)

    # Pickup & delivery pairing (mandatory)
    for i in range(len(requests)):
        p_node = request_to_pickup[i]
        d_node = request_to_dropoff[i]
        p_idx = manager.NodeToIndex(p_node)
        d_idx = manager.NodeToIndex(d_node)

        routing.AddPickupAndDelivery(p_idx, d_idx)
        # same vehicle
        routing.solver().Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
        # pickup before dropoff in time
        time_dimension.SetCumulVarSoftLowerBound(d_idx, 0, 0)  # no penalty, just initialize var
        routing.solver().Add(time_dimension.CumulVar(p_idx) <= time_dimension.CumulVar(d_idx))

    # Search parameters
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # Support fractional seconds for time limit
    _max_time = max(0.0, float(max_solve_time))
    _whole = int(_max_time)
    _nanos = int(round((_max_time - _whole) * 1e9))
    search.time_limit.seconds = _whole
    # OR-Tools Duration supports nanos attribute
    try:
        search.time_limit.nanos = _nanos
    except Exception:
        # Fallback for older OR-Tools builds without nanos field
        pass
    search.log_search = False

    solution = routing.SolveWithParameters(search)

    if solution is None:
        return {
            "status": "INFEASIBLE",
            "total_distance_km": 0.0,
            "num_vehicles_used": 0,
            "routes": [],
        }

    # Extract routes and distance
    routes = []
    total_distance_m = 0
    for v in range(num_vehicles):
        route_nodes = []
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            # empty vehicle, skip adding an empty route
            continue
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot:
                route_nodes.append(node)
            next_index = solution.Value(routing.NextVar(index))
            total_distance_m += routing.GetArcCostForVehicle(index, next_index, v)
            index = next_index
        if route_nodes:
            routes.append(route_nodes)

    return {
        "status": "FEASIBLE",
        "total_distance_km": total_distance_m / 1000.0,
        "num_vehicles_used": len(routes),
        "routes": routes,
    }


# --------------------------- Helper functions --------------------------- #

def _create_nodes(requests):
    """Create pickup and dropoff node ids for each request."""
    pickup_nodes = []
    dropoff_nodes = []
    node_to_request = {}
    request_to_pickup = {}
    request_to_dropoff = {}

    # Depot is node 0; we will index pickups 1..N and dropoffs N+1..2N
    depot = 0
    next_node = 1

    for i, req in enumerate(requests):
        p_node = next_node
        d_node = next_node + 1
        next_node += 2
        pickup_nodes.append(p_node)
        dropoff_nodes.append(d_node)
        node_to_request[p_node] = i
        node_to_request[d_node] = i
        request_to_pickup[i] = p_node
        request_to_dropoff[i] = d_node

    return pickup_nodes, dropoff_nodes, node_to_request, request_to_pickup, request_to_dropoff


def _get_node_location(node, depot_node, pickup_nodes, dropoff_nodes, node_to_request, requests):
    """Map a solver node id back to a location index in the distance matrix.

    Returns the index into the distance matrix corresponding to the given node.
    """
    # Depot maps to depot_node
    if node == 0:
        return depot_node

    # Pickup nodes map to the origin location of the corresponding request
    if node in pickup_nodes:
        req_idx = node_to_request.get(node)
        if req_idx is None:
            return None
        return requests[req_idx]["origin"]

    # Dropoff nodes map to the destination location of the corresponding request
    if node in dropoff_nodes:
        req_idx = node_to_request.get(node)
        if req_idx is None:
            return None
        return requests[req_idx]["destination"]

    return None


if __name__ == "__main__":
    # Minimal demo (requires your own create_network/simulation if you want to run end-to-end)
    import pprint

    # Tiny synthetic example (3 requests, 1 depot + 6 nodes -> 7x7)
    requests, index_list = sim.simulation(N=5)
    enc_net = nodify.create_network(requests)
    dm = enc_net['distance']

    reqs = enc_net['requests']

    res = cost_estimator( 
        distance_matrix=dm,
        requests=reqs,
        vehicle_num=1,
        depot_node=0,
        vehicle_travel_speed=20.0,
        vehicle_penalty=500.0,
        time_window_duration=30,
        vehicle_capacity=2,
        max_vehicles=3,
        max_solve_time=10,
    )
    print(requests)
    print(enc_net['map'])
    pprint.pprint(res)
