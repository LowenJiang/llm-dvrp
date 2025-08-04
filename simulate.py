
import numpy as np
import random
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def generate_darp_instance(n_requests=20, coord_range=100, min_pickup_time=300, max_pickup_time=1000, pickup_time_window=30, velocity=1, random_seed=None):
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Generate coordinates for pickup and dropoff locations
    def generate_coordinates(n_requests):
        coordinates = []
        for i in range(n_requests):
            # Generate pickup coordinates
            p1 = random.randint(0, coord_range)
            p2 = random.randint(0, coord_range)
            pickup_quad = (0 if p1 <= coord_range//2 else 1, 0 if p2 <= coord_range//2 else 1)

            # Generate dropoff in different quadrant
            while True:
                d1 = random.randint(0, coord_range)
                d2 = random.randint(0, coord_range)
                dropoff_quad = (0 if d1 <= coord_range//2 else 1, 0 if d2 <= coord_range//2 else 1)
                if pickup_quad != dropoff_quad:
                    break

            coordinates.append(((p1, p2), (d1, d2)))
        return coordinates

    # Generate time windows
    def generate_time_windows(coordinates):
        time_windows = []
        for (p1, p2), (d1, d2) in coordinates:
            pickup_start = random.randint(min_pickup_time, max_pickup_time)
            pickup_end = pickup_start + pickup_time_window
            
            # Only return pickup time windows - dropoff will be dynamically constrained
            time_windows.append((pickup_start, pickup_end))
        return time_windows

    # Create distance matrix (Euclidean, integer)
    def create_distance_matrix(coordinates):
        # Depot at center of coordinate range
        depot_x = coord_range // 2
        depot_y = coord_range // 2
        all_locations = [(depot_x, depot_y)]  # depot at center
        for (p1, p2), (d1, d2) in coordinates:
            all_locations.append((p1, p2))  # pickup
            all_locations.append((d1, d2))  # dropoff

        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
        for i in range(n_locations):
            x1, y1 = all_locations[i]
            for j in range(n_locations):
                if i == j:
                    continue
                x2, y2 = all_locations[j]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distance_matrix[i][j] = int(np.ceil(dist))  # round up to ensure feasibility
        return distance_matrix

    # Generate problem data
    coordinates = generate_coordinates(n_requests)
    time_windows = generate_time_windows(coordinates)
    distance_matrix = create_distance_matrix(coordinates)

    return {"coordinates": coordinates, "time_windows": time_windows, "distance_matrix": distance_matrix}


def create_darp_model(coordinates, time_windows, distance_matrix, num_vehicles, time_limit, coord_range=100):
    n_requests = len(coordinates)

    depot_index = 0  # Depot is always at index 0 in the distance matrix
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

    # Add capacity dimension for personnel constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:  # depot
            return 0
        elif from_node % 2 == 1:  # pickup nodes (odd numbers)
            return 1  # +1 person at pickup
        else:  # dropoff nodes (even numbers)
            return -1  # -1 person at dropoff

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [5] * num_vehicles,  # vehicle maximum capacities (5 people per vehicle)
        True,  # start cumul to zero
        'Capacity'
    )
    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    # Apply time window constraints only if time_windows is provided
    if time_windows is not None:
        # Build full list of time windows: depot + pickups + dynamic dropoffs
        time_windows_list = [(0, horizon)]  # depot
        
        for request_idx, pickup_window in enumerate(time_windows):
            pickup_start, pickup_end = pickup_window
            
            # Add pickup time window
            time_windows_list.append((pickup_start, pickup_end))
            
            # Calculate dynamic dropoff time window based on pickup + travel constraints
            pickup_node = 2 * request_idx + 1
            dropoff_node = 2 * request_idx + 2
            travel_time = distance_matrix[pickup_node][dropoff_node]
            
            # Dropoff must be between pickup + 1x travel_time to pickup + 2x travel_time
            # For any pickup time in [pickup_start, pickup_end], calculate valid dropoff range
            dropoff_earliest = pickup_start + travel_time      # earliest pickup + min travel
            dropoff_latest = pickup_end + 2 * travel_time      # latest pickup + max travel
            
            time_windows_list.append((dropoff_earliest, dropoff_latest))

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
        
        # Constraint: dropoff_time >= pickup_time + 1x travel_time (minimum)
        routing.solver().Add(
            time_dimension.CumulVar(dropoff_index) >= time_dimension.CumulVar(pickup_index) + travel_time
        )
        
        # Constraint: dropoff_time <= pickup_time + 2x travel_time (maximum)
        routing.solver().Add(
            time_dimension.CumulVar(dropoff_index) <= time_dimension.CumulVar(pickup_index) + 2 * travel_time
        )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit 
    search_parameters.log_search = False  # Disable solution logging

    return manager, routing, time_dimension, search_parameters




def print_solution(manager, routing, time_dimension, solution):
    if not solution:
        print("No solution found.")
        return

    print("Solution found!")
    total_distance = 0
    total_time = 0
    num_served = 0
    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        route_repr = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            time_val = solution.Value(time_var)
            capacity_var = capacity_dimension.CumulVar(index)
            capacity_val = solution.Value(capacity_var)
            route_repr.append((node, time_val, capacity_val))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            # Add cost for ALL arcs, including the final return to depot
            total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # end node
        end_node = manager.IndexToNode(index)
        end_time = solution.Value(time_dimension.CumulVar(index))
        end_capacity = solution.Value(capacity_dimension.CumulVar(index))
        route_repr.append((end_node, end_time, end_capacity))
        print(f"Vehicle {vehicle_id} route (node, time, passengers): {route_repr}")
        num_pickups = sum(1 for node, _, _ in route_repr if node > 0 and node % 2 == 1)
        num_served += num_pickups

    print(f"Total distance (cost): {total_distance}")
    print(f"Number of served requests (pickup/dropoff pairs): {num_served}")

def simulate(n_requests=30, coord_range=100, min_pickup_time=300, max_pickup_time=1000, pickup_time_window=30, velocity=1, num_vehicles=10, time_limit=10, use_time_windows=True, random_seed=None):
    instance = generate_darp_instance(n_requests, coord_range, min_pickup_time, max_pickup_time, pickup_time_window, velocity, random_seed)
    
    # Use time windows only if requested
    time_windows = instance["time_windows"] if use_time_windows else None
    
    print(f"Solving DARP with {n_requests} requests, {num_vehicles} vehicles, time limit: {time_limit}s, time windows: {'enabled' if use_time_windows else 'disabled'}")
    manager, routing, time_dimension, search_parameters = create_darp_model(instance["coordinates"], time_windows, instance["distance_matrix"], num_vehicles, time_limit, coord_range)
    
    # First solve with time limit
    solution = routing.SolveWithParameters(search_parameters)
    
    # Print solution status and time for first attempt
    if solution:
        print(f"Solution found in {routing.solver().WallTime():.3f} seconds")
    else:
        print(f"No solution found within {time_limit} seconds")
    
    print_solution(manager, routing, time_dimension, solution)
    return {"solution": solution, "manager": manager, "routing": routing, "time_dimension": time_dimension, "search_parameters": search_parameters}

if __name__ == "__main__":
    simulate()