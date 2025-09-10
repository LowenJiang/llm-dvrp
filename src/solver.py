"""
DARP (Dial-a-Ride Problem) Solver using OR-Tools.

This module implements a solver for the Dial-a-Ride Problem, which involves:
- Picking up customers at their requested time windows
- Dropping them off at their destinations within time windows
- Optimizing vehicle routes to minimize total travel time/distance
- Handling vehicle capacity constraints
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import Dict, List, Tuple, Optional
import time


class DARPSolver:
    """
    Solver for the Dial-a-Ride Problem using OR-Tools.
    
    The solver handles:
    - Multiple vehicles with capacity constraints
    - Pickup and dropoff time windows
    - Request pairing (pickup before dropoff)
    - Distance matrix optimization
    """
    
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 requests: List[Dict],
                 map_data: Dict,
                 num_vehicles: int = 3,
                 vehicle_capacity: int = 4,
                 time_window_duration: int = 30):
        """
        Initialize the DARP solver.
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Distance matrix between all locations (in km)
        requests : List[Dict]
            List of requests with 'origin', 'destination', 'o_t_index', 'd_t_index'
        map_data : Dict
            Mapping from spatial indices to H3 indices
        num_vehicles : int, default=3
            Number of available vehicles
        vehicle_capacity : int, default=4
            Maximum capacity per vehicle (number of passengers)
        time_window_duration : int, default=30
            Duration of each time window in minutes
        """
        self.distance_matrix = distance_matrix
        self.requests = requests
        self.map_data = map_data
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.time_window_duration = time_window_duration
        
        # Create pickup and dropoff nodes
        self._create_nodes()
        
        # Create distance callback
        self.distance_callback = self._create_distance_callback()
        
        # Create demand callback for capacity constraints
        self.demand_callback = self._create_demand_callback()
        
        # Create time callback for time window constraints
        self.time_callback = self._create_time_callback()
    
    def _create_nodes(self):
        """Create pickup and dropoff nodes for each request."""
        self.pickup_nodes = []
        self.dropoff_nodes = []
        self.node_to_request = {}
        self.request_to_pickup = {}
        self.request_to_dropoff = {}
        
        node_index = 0
        
        # Add depot node (index 0)
        self.depot = 0
        node_index += 1
        
        # Create pickup and dropoff nodes for each request
        for i, request in enumerate(self.requests):
            # Pickup node
            pickup_node = node_index
            self.pickup_nodes.append(pickup_node)
            self.node_to_request[pickup_node] = (i, 'pickup')
            self.request_to_pickup[i] = pickup_node
            node_index += 1
            
            # Dropoff node
            dropoff_node = node_index
            self.dropoff_nodes.append(dropoff_node)
            self.node_to_request[dropoff_node] = (i, 'dropoff')
            self.request_to_dropoff[i] = dropoff_node
            node_index += 1
        
        self.num_nodes = node_index
        self.num_requests = len(self.requests)
    
    def _create_distance_callback(self):
        """Create distance callback for OR-Tools."""
        def distance_callback(from_index, to_index):
            from_node = self._index_to_node(from_index)
            to_node = self._index_to_node(to_index)
            
            if from_node == to_node:
                return 0
            
            # Get actual locations from the nodes
            from_location = self._get_node_location(from_node)
            to_location = self._get_node_location(to_node)
            
            if from_location is None or to_location is None:
                return 0
            
            return int(self.distance_matrix[from_location][to_location] * 1000)  # Convert to meters
        
        return distance_callback
    
    def _create_demand_callback(self):
        """Create demand callback for capacity constraints."""
        def demand_callback(from_index):
            from_node = self._index_to_node(from_index)
            
            # Depot has no demand
            if from_node == self.depot:
                return 0
            
            # Pickup nodes have demand +1, dropoff nodes have demand -1
            if from_node in self.pickup_nodes:
                return 1
            elif from_node in self.dropoff_nodes:
                return -1
            
            return 0
        
        return demand_callback
    
    def _create_time_callback(self):
        """Create time callback for time window constraints."""
        def time_callback(from_index, to_index):
            from_node = self._index_to_node(from_index)
            to_node = self._index_to_node(to_index)
            
            if from_node == to_node:
                return 0
            
            # Get actual locations from the nodes
            from_location = self._get_node_location(from_node)
            to_location = self._get_node_location(to_node)
            
            if from_location is None or to_location is None:
                return 0
            
            # Travel time in minutes (assuming 20 km/h average speed)
            distance_km = self.distance_matrix[from_location][to_location]
            travel_time_minutes = int(distance_km / 20 * 60)  # 20 km/h = 1/3 km/min
            
            return travel_time_minutes
        
        return time_callback
    
    def _index_to_node(self, index):
        """Convert OR-Tools index to our node index."""
        if index == 0:
            return self.depot
        return index
    
    def _node_to_index(self, node):
        """Convert our node index to OR-Tools index."""
        if node == self.depot:
            return 0
        return node
    
    def _get_node_location(self, node):
        """Get the spatial location index for a node."""
        if node == self.depot:
            # Depot is at the first location (index 0) - center of SF
            return 0
        
        if node in self.pickup_nodes:
            request_idx, _ = self.node_to_request[node]
            return self.requests[request_idx]['origin']
        elif node in self.dropoff_nodes:
            request_idx, _ = self.node_to_request[node]
            return self.requests[request_idx]['destination']
        
        return None
    
    def _make_requests_optional(self, routing, manager):
        """Make all requests optional so some can be dropped if infeasible."""
        for i in range(self.num_requests):
            pickup_node = self.request_to_pickup[i]
            dropoff_node = self.request_to_dropoff[i]
            
            pickup_index = manager.NodeToIndex(pickup_node)
            dropoff_index = manager.NodeToIndex(dropoff_node)
            
            # Make both pickup and dropoff optional with a penalty cost
            # Higher penalty for dropoff to encourage keeping requests
            pickup_penalty = 1000  # Cost to skip pickup
            dropoff_penalty = 2000  # Cost to skip dropoff
            
            routing.AddDisjunction([pickup_index], pickup_penalty)
            routing.AddDisjunction([dropoff_index], dropoff_penalty)

    def solve(self, time_limit_seconds: int = 30) -> Dict:
        """
        Solve the DARP problem.
        
        Parameters:
        -----------
        time_limit_seconds : int, default=30
            Maximum time to spend solving in seconds
        
        Returns:
        --------
        Dict
            Solution containing routes, total distance, and other metrics
        """
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            self.num_nodes, self.num_vehicles, self.depot
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Add distance callback
        transit_callback_index = routing.RegisterTransitCallback(self.distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints
        demand_callback_index = routing.RegisterUnaryTransitCallback(self.demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.vehicle_capacity] * self.num_vehicles,  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add time window constraints
        time_callback_index = routing.RegisterTransitCallback(self.time_callback)
        routing.AddDimension(
            time_callback_index,
            30,  # allow waiting time
            1440,  # maximum time per day (24 hours in minutes)
            False,  # don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Set time windows for pickup and dropoff nodes
        for i, request in enumerate(self.requests):
            pickup_node = self.request_to_pickup[i]
            dropoff_node = self.request_to_dropoff[i]
            
            # Pickup time window
            pickup_start = request['o_t_index'] * self.time_window_duration
            pickup_end = pickup_start + self.time_window_duration
            time_dimension.CumulVar(manager.NodeToIndex(pickup_node)).SetRange(
                pickup_start, pickup_end
            )
            
            # Dropoff time window
            dropoff_start = request['d_t_index'] * self.time_window_duration
            dropoff_end = dropoff_start + self.time_window_duration
            time_dimension.CumulVar(manager.NodeToIndex(dropoff_node)).SetRange(
                dropoff_start, dropoff_end
            )
        
        # Make requests optional (allow dropping some)
        self._make_requests_optional(routing, manager)
        
        # Add pickup-dropoff precedence constraints using the correct OR-Tools syntax
        for i in range(self.num_requests):
            pickup_node = self.request_to_pickup[i]
            dropoff_node = self.request_to_dropoff[i]
            
            pickup_index = manager.NodeToIndex(pickup_node)
            dropoff_index = manager.NodeToIndex(dropoff_node)
            
            # Both pickup and dropoff must be served by the same vehicle
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
            )
            
            # Pickup must happen before dropoff - use time dimension constraints
            pickup_time = time_dimension.CumulVar(pickup_index)
            dropoff_time = time_dimension.CumulVar(dropoff_index)
            
            # Add constraint: pickup_time <= dropoff_time
            routing.solver().Add(pickup_time <= dropoff_time)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds
        search_parameters.log_search = True
        
        # Solve the problem
        start_time = time.time()
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
        if solution is None:
            return {
                'status': 'OPTIMAL' if solution else 'INFEASIBLE',
                'routes': [],
                'total_distance': 0,
                'solve_time': solve_time,
                'unassigned_requests': list(range(self.num_requests)),
                'num_vehicles_used': 0
            }
        routes = []
        total_distance = 0
        unassigned_requests = []
        
        for vehicle_id in range(self.num_vehicles):
            route = []
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != self.depot:
                    route.append(node)
                index = solution.Value(routing.NextVar(index))
            
            if route:
                routes.append(route)
                # Calculate route distance
                route_distance = 0
                prev_index = routing.Start(vehicle_id)
                for node in route:
                    current_index = manager.NodeToIndex(node)
                    route_distance += solution.Value(
                        routing.GetArcCostForVehicle(prev_index, current_index, vehicle_id)
                    )
                    prev_index = current_index
                total_distance += route_distance
        
        # Find unassigned requests
        assigned_requests = set()
        for route in routes:
            for node in route:
                if node in self.pickup_nodes:
                    request_idx, _ = self.node_to_request[node]
                    assigned_requests.add(request_idx)
        
        unassigned_requests = [
            i for i in range(self.num_requests) 
            if i not in assigned_requests
        ]
        
        return {
            'status': 'OPTIMAL' if solution else 'INFEASIBLE',
            'routes': routes,
            'total_distance': total_distance / 1000,  # Convert back to km
            'solve_time': solve_time,
            'unassigned_requests': unassigned_requests,
            'num_vehicles_used': len(routes)
        }
    
    def get_route_details(self, solution: Dict) -> List[Dict]:
        """
        Get detailed information about each route.
        
        Parameters:
        -----------
        solution : Dict
            Solution from solve() method
        
        Returns:
        --------
        List[Dict]
            List of route details with pickup/dropoff information
        """
        route_details = []
        
        for i, route in enumerate(solution['routes']):
            route_info = {
                'vehicle_id': i,
                'stops': [],
                'total_distance_km': 0,
                'requests_served': []
            }
            
            for node in route:
                if node in self.pickup_nodes:
                    request_idx, _ = self.node_to_request[node]
                    request = self.requests[request_idx]
                    route_info['stops'].append({
                        'type': 'pickup',
                        'request_id': request_idx,
                        'location': request['origin'],
                        'time_window': request['o_t_index']
                    })
                    route_info['requests_served'].append(request_idx)
                elif node in self.dropoff_nodes:
                    request_idx, _ = self.node_to_request[node]
                    request = self.requests[request_idx]
                    route_info['stops'].append({
                        'type': 'dropoff',
                        'request_id': request_idx,
                        'location': request['destination'],
                        'time_window': request['d_t_index']
                    })
            
            route_details.append(route_info)
        
        return route_details


def solve_darp(distance_matrix: np.ndarray, 
               requests: List[Dict], 
               map_data: Dict,
               num_vehicles: int = 10,
               vehicle_capacity: int = 4,
               time_limit_seconds: int = 30) -> Dict:
    """
    Convenience function to solve DARP problem.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix between all locations
    requests : List[Dict]
        List of requests with pickup/dropoff information
    map_data : Dict
        Mapping from spatial indices to H3 indices
    num_vehicles : int, default=3
        Number of available vehicles
    vehicle_capacity : int, default=4
        Maximum capacity per vehicle
    time_limit_seconds : int, default=30
        Maximum solve time in seconds
    
    Returns:
    --------
    Dict
        Solution with routes, distances, and metrics
    """
    solver = DARPSolver(
        distance_matrix=distance_matrix,
        requests=requests,
        map_data=map_data,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    
    solution = solver.solve(time_limit_seconds)
    solution['route_details'] = solver.get_route_details(solution)
    
    return solution




def solve_darp_iterative(distance_matrix: np.ndarray, 
                        requests: List[Dict], 
                        map_data: Dict,
                        initial_vehicles: int = 1,
                        max_vehicles: int = 10,
                        vehicle_capacity: int = 4,
                        time_limit_seconds: int = 30,
                        vehicle_penalty: int = 1000) -> Dict:
    """
    Iteratively solve DARP by adding vehicles when infeasible.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix between all locations
    requests : List[Dict]
        List of requests with pickup/dropoff information
    map_data : Dict
        Mapping from spatial indices to H3 indices
    initial_vehicles : int, default=1
        Starting number of vehicles
    max_vehicles : int, default=10
        Maximum number of vehicles to try
    vehicle_capacity : int, default=4
        Maximum capacity per vehicle
    time_limit_seconds : int, default=30
        Maximum solve time per attempt in seconds
    vehicle_penalty : int, default=1000
        Penalty cost for using an additional vehicle
    
    Returns:
    --------
    Dict
        Solution with routes, distances, and metrics
    """
    best_solution = None
    best_cost = float('inf')
    
    for num_vehicles in range(initial_vehicles, max_vehicles + 1):
        print(f'
Trying with {num_vehicles} vehicles...')
        
        # Solve with current number of vehicles
        solution = solve_darp(
            distance_matrix=distance_matrix,
            requests=requests,
            map_data=map_data,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            time_limit_seconds=time_limit_seconds
        )
        
        # Calculate total cost including vehicle penalty
        total_cost = solution['total_distance'] + (num_vehicles * vehicle_penalty)
        
        print(f'  Status: {solution["status"]}')
        print(f'  Distance: {solution["total_distance"]:.2f} km')
        print(f'  Vehicles used: {solution["num_vehicles_used"]}')
        print(f'  Unassigned requests: {len(solution["unassigned_requests"])}')
        print(f'  Total cost: {total_cost:.2f}')
        
        # Check if this is a better solution
        if solution['status'] == 'OPTIMAL' and total_cost < best_cost:
            best_solution = solution.copy()
            best_solution['total_cost'] = total_cost
            best_solution['num_vehicles_attempted'] = num_vehicles
            best_cost = total_cost
            
            # If all requests are served, we found the optimal solution
            if len(solution['unassigned_requests']) == 0:
                print(f'  ✓ All requests served with {num_vehicles} vehicles!')
                break
            else:
                print(f'  ✓ Better solution found, but {len(solution["unassigned_requests"])} requests still unassigned')
        else:
            print(f'  ✗ No improvement with {num_vehicles} vehicles')
    
    if best_solution is None:
        # Fallback: return the last attempted solution
        best_solution = solution
        best_solution['total_cost'] = total_cost
        best_solution['num_vehicles_attempted'] = num_vehicles
    
    return best_solution


if __name__ == "__main__":
    # Example usage
    from nodify import create_network
    from sim import simulation
    
    # Generate sample data
    requests_df = simulation(N=5)
    network_data = create_network(requests_df)
    print(network_data['requests'])
    # Solve DARP
    solution = solve_darp(
        distance_matrix=network_data['distance'],
        requests=network_data['requests'],
        map_data=network_data['map'],
        num_vehicles=5,
        vehicle_capacity=3
    )
    
    print("DARP Solution:")
    print(f"Status: {solution['status']}")
    print(f"Total Distance: {solution['total_distance']:.2f} km")
    print(f"Solve Time: {solution['solve_time']:.2f} seconds")
    print(f"Vehicles Used: {solution['num_vehicles_used']}")
    print(f"Unassigned Requests: {solution['unassigned_requests']}")
    
    for i, route_detail in enumerate(solution['route_details']):
        print(f"\nVehicle {i} Route:")
        for stop in route_detail['stops']:
            print(f"  {stop['type'].upper()}: Request {stop['request_id']} at location {stop['location']} (time window {stop['time_window']})")
