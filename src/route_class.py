import sim, solver, nodify
import numpy as np

class Route: 
    """
    A class to represent a route. 
    It contains the following attributes:
    - N: number of requests
    - vehicle_num: number of vehicles
    - vehicle_penalty: penalty for using a vehicle
    - max_vehicles: maximum number of vehicles
    - vehicle_speed: speed of vehicles
    - depot_node: depot node
    - time_window_duration: time window duration
    - vehicle_capacity: capacity of vehicles
    - auto_solve: whether to automatically solve the route
    - max_vehicles: maximum number of vehicles
    - auto_solve: whether to automatically solve the route
    - max_solve_time: maximum time for solving the route

    It also contains the following methods:
    - print_solution: print the solution
    - solve: solve the route
    - print_cost: print the cost
    - print_solution: print the solution
    - aggregation: return 4D tensor using current requests and h3 indices
    - modify: modify a request's time windows and update aggregation

    The final solution has the following structure:
    'status': 'FEASIBLE' or 'INFEASIBLE'
    'total_distance_km': total distance of the route
    'num_vehicles_used': number of vehicles used
    'routes': list of routes, each route is a list of nodes
    'routing_cost': same as 'total_distance_km' in the simplified version
    'vehicle_penalty_cost': extra cost incurred by introducing another vehicle (when INFEASIBLE, iteratively increase the number of vehicles till FEASIBLE)
    ---'total_cost': total cost of the route. This is the sum of 'total_distance_km' and 'vehicle_penalty_cost', and the real value we are interested in. 
    'num_vehicles_attempted': as the name suggests
    'solve_time': as the name suggests
    """
    def __init__(self, N=20, 
                 vehicle_num=4, 
                 vehicle_penalty=500.0,
                 max_vehicles=10,
                 vehicle_speed = 20.0,
                 depot_node = 0,
                 time_window_duration = 30,
                 vehicle_capacity = 4,
                 auto_solve = False,
                 max_solve_time = 5):
        
        self.N = N
        self.vehicle_num = vehicle_num
        self.vehicle_penalty = vehicle_penalty
        self.max_vehicles = max_vehicles
        self.vehicle_speed = vehicle_speed
        self.time_window_duration = time_window_duration
        self.vehicle_capacity = vehicle_capacity
        self.depot_node = depot_node
        self.max_solve_time = max_solve_time
        self.auto_solve = auto_solve
        
        # sim.simulation returns (requests_df, sf_h3_indices)
        self.requests_df, self.loc_indices = sim.simulation(N=N, vehicle_speed_kmh=self.vehicle_speed)
        self.enc_net = nodify.create_network(self.requests_df)
        self.dm = self.enc_net['distance']
        self.reqs = self.enc_net['requests']
        
        # Initialize cost as None
        self.cost = None

        if self.auto_solve:
            self.cost = self.solve()

    def print_solution(self):
        if self.cost is not None:
            print(self.cost)
        else:
            print("Cost not computed yet")

    def solve(self):
        self.cost = solver.cost_estimator(
            distance_matrix=self.dm,
            requests=self.reqs,
            vehicle_num=self.vehicle_num,
            vehicle_penalty=self.vehicle_penalty,
            max_vehicles=self.max_vehicles,
            vehicle_travel_speed=self.vehicle_speed,
            time_window_duration=self.time_window_duration,
            vehicle_capacity=self.vehicle_capacity,
            depot_node=self.depot_node,
            max_solve_time=self.max_solve_time
        )
        return self.cost['total_cost']

    def aggregation(self):
        """
        Return a 4D tensor using the current Route requests and the list of h3 indices.
        Load self.M with the aggregate counts.
        
        Returns:
        --------
        np.ndarray: 4D tensor of shape (ℓ, ℓ, h, h)
        """
        # Get the list of H3 indices
        h3_indices = self.loc_indices
        
        # Initialize 4D tensor
        ℓ = len(h3_indices)
        h = 48  # Assuming 48 time intervals (24h / 30min)
        self.M = np.zeros((ℓ, ℓ, h, h), dtype=np.int32)
        
        # Aggregate current requests
        for req in self.reqs:  # self.reqs contains the processed requests
            # req['origin'] and req['destination'] are spatial indices (0 to ℓ-1)
            # that correspond to positions in self.loc_indices
            o_region = req['origin']  # Already a region index
            d_region = req['destination']  # Already a region index
            
            # Convert time indices (assuming 1-indexed in requests, convert to 0-indexed)
            o_t = req['o_t_index'] - 1
            d_t = req['d_t_index'] - 1
            
            # Ensure indices are within bounds
            o_t = max(0, min(h-1, o_t))
            d_t = max(0, min(h-1, d_t))
            
            # Update aggregate tensor
            self.M[o_region, d_region, o_t, d_t] += 1
        
        return self.M

    def modify(self, delta1, delta2, index):
        """
        Modify the (index)th Route request's o_t_index and d_t_index by delta1 and delta2 respectively.
        Run the aggregation function to update self.M.
        
        Parameters:
        -----------
        delta1: int - offset for o_t_index
        delta2: int - offset for d_t_index  
        index: int - index of request to modify
        """
        # Check if index is valid
        if index < 0 or index >= len(self.reqs):
            raise ValueError(f"Invalid request index {index}. Valid range: 0-{len(self.reqs)-1}")
        
        # Modify the request
        self.reqs[index]['o_t_index'] += delta1
        self.reqs[index]['d_t_index'] += delta2
        
        # Ensure feasibility constraints
        self.reqs[index]['o_t_index'] = max(1, self.reqs[index]['o_t_index'])  # At least 1
        self.reqs[index]['d_t_index'] = max(1, self.reqs[index]['d_t_index'])  # At least 1
        self.reqs[index]['o_t_index'] = min(48, self.reqs[index]['o_t_index'])  # At most 48
        self.reqs[index]['d_t_index'] = min(48, self.reqs[index]['d_t_index'])  # At most 48
        
        # Ensure pickup before dropoff
        if self.reqs[index]['o_t_index'] >= self.reqs[index]['d_t_index']:
            self.reqs[index]['d_t_index'] = self.reqs[index]['o_t_index'] + 1
        
        # Update aggregation
        self.aggregation()
        
        # Re-solve the route
        self.solve()

if __name__ == "__main__":
    route = Route()
    route.print_solution()
    print(route.requests_df)  # Print the dataframe
    
    # Test the new functions
    print("\nTesting aggregation function:")
    M = route.aggregation()
    print(f"Aggregate tensor shape: {M.shape}")
    print(f"Non-zero entries: {np.count_nonzero(M)}")
    
    print("\nTesting modify function:")
    print(f"Original request 0: {route.reqs[0]}")

    print("cost before modify: ", route.solve())

    route.modify(-5, 5, 0)  # Modify first request
    route.solve()
    print(f"Modified request 0: {route.reqs[0]}")
    print("cost after modify: ", route.solve())

