import sim, solver, nodify

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
    - max_solve_time: maximum time for solving the route

    It also contains the following methods:
    - print_solution: print the solution
    - solve: solve the route
    - print_cost: print the cost
    - print_solution: print the solution

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
                 max_vehicles=6,
                 vehicle_speed = 20.0,
                 depot_node = 0,
                 time_window_duration = 30,
                 vehicle_capacity = 4,
                 auto_solve = True,
                 max_solve_time = 3):
        
        self.N = N
        self.vehicle_num = vehicle_num
        self.vehicle_penalty = vehicle_penalty
        self.max_vehicles = max_vehicles
        self.vehicle_speed = vehicle_speed
        self.time_window_duration = time_window_duration
        self.vehicle_capacity = vehicle_capacity
        self.depot_node = depot_node

        self.requests = sim.simulation(N=N, vehicle_speed_kmh=self.vehicle_speed)
        self.enc_net = nodify.create_network(self.requests)
        self.dm = self.enc_net['distance']
        self.reqs = self.enc_net['requests']

        if auto_solve:
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

if __name__ == "__main__":
    route = Route()
    route.print_solution()