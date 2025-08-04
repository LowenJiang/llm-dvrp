import numpy as np
import random
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_darp_model(coordinates, time_windows, distance_matrix, num_vehicles=5, time_limit=1):
    n_requests = len(coordinates)
    depot_index = 0
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        num_vehicles,
        depot_index
    )
    routing = pywrapcp.RoutingModel(manager)

    # Distance (and travel time) callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time dimension: travel time is distance, allow waiting
    horizon = 5000
    routing.AddDimension(
        transit_callback_index,
        2000,      # slack / allow waiting
        horizon,   # maximum time per vehicle
        True,      # force start cumul to zero
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')

    # Build full list of time windows: depot + pickups + dropoffs
    time_windows_list = [(0, horizon)]  # depot
    for tw_pickup, tw_dropoff in time_windows:
        time_windows_list.append(tw_pickup)
        time_windows_list.append(tw_dropoff)

    # Apply time window constraints
    for loc_idx, (start, end) in enumerate(time_windows_list):
        index = manager.NodeToIndex(loc_idx)
        time_dimension.CumulVar(index).SetRange(int(start), int(end))

    # Pickup and delivery pairing with travel-time enforcement
    for request_idx in range(n_requests):
        pickup_node = 2 * request_idx + 1
        dropoff_node = 2 * request_idx + 2
        pickup_index = manager.NodeToIndex(pickup_node)
        dropoff_index = manager.NodeToIndex(dropoff_node)

        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
        )
        # Enforce that dropoff occurs after pickup plus travel time
        travel_time = distance_matrix[pickup_node][dropoff_node]
        routing.solver().Add(
            time_dimension.CumulVar(dropoff_index) >= time_dimension.CumulVar(pickup_index) + travel_time
        )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = False  # Disable solution logging

    return manager, routing, time_dimension, search_parameters

