import numpy as np
from typing import Tuple, Dict, List, Optional
from route_class import Route


class DVRPEnvironment:
    """
    Streamlined DVRP Environment using Route class
    
    This environment uses the Route class as input and leverages its
    aggregation() and modify() functions for state management.
    """
    
    def __init__(self, 
                 route: Route,
                 acceptance_rate: float = 0.5,
                 alpha: float = 2.0):
        
        self.route = route
        self.acceptance_rate = acceptance_rate
        self.alpha = alpha
        self.current_request_index = 0
        self.episode_step = 0
        
        # Initialize aggregation
        self.M = self.route.aggregation()
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and start new episode
        
        Returns:
        --------
        state: np.ndarray
            Initial state tensor of shape (ℓ, ℓ, h, h)
        request: Dict
            Initial request with keys: origin, destination, o_t_index, d_t_index
        """
        self.current_request_index = 0
        self.episode_step = 0
        
        # Re-initialize aggregation
        self.M = self.route.aggregation()
        self.route.solve()
        
        return self.M.copy(), self._get_current_request()
    
    def _get_current_request(self) -> Dict:
        """Get current request information"""
        if self.current_request_index < len(self.route.reqs):
            req = self.route.reqs[self.current_request_index]
            return {
                'origin': req['origin'],
                'destination': req['destination'],
                'o_t_index': req['o_t_index'],
                'd_t_index': req['d_t_index']
            }
        else:
            return None
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Get valid actions for current request
        
        Returns:
        --------
        List of valid (δ^o, δ^d) tuples such that the modified pickup time index ≥ 1,
        the modified dropoff time index ≤ 48, and modified pickup ≤ modified dropoff.
        """
        if self.current_request_index >= len(self.route.reqs):
            return []
        
        valid_actions = []
        req = self.route.reqs[self.current_request_index]
        o_t = req['o_t_index']
        d_t = req['d_t_index']
        
        # Generate all possible time window offsets
        for delta_o in range(-47, 48):  # -47 to 47
            for delta_d in range(-47, 48):  # -47 to 47
                new_o_t = o_t + delta_o
                new_d_t = d_t + delta_d
                
                # Check feasibility constraints
                if (new_o_t >= 1 and
                    new_d_t <= 48 and
                    new_o_t <= new_d_t):
                    valid_actions.append((delta_o, delta_d))
        
        return valid_actions
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, Dict, float, bool, Dict]:
        """
        Execute action and return next state
        
        Parameters:
        -----------
        action: Tuple[int, int]
            Action (δ^o, δ^d) to execute
            
        Returns:
        --------
        next_state: np.ndarray
            Updated aggregate tensor
        next_request: Dict
            Next request in episode
        reward: float
            Step-wise reward equal to Route.solve() cost improvement,
            with local penalty alpha if action is rejected.
        done: bool
            Whether episode is finished
        info: Dict
            Additional information including acceptance and cost info
        """
        delta_o, delta_d = action
        
        # Check if action is valid
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            raise ValueError(f"Invalid action {action}. Valid actions: {len(valid_actions)} available")
        
        # Ensure we have a baseline cost for this state
        if not self.route.cost or 'total_cost' not in self.route.cost:
            self.route.solve()
        
        # Get cost before action
        cost_before = self.route.cost.get('total_cost', float('inf'))
        if np.isnan(cost_before) or np.isinf(cost_before):
            cost_before = 1000.0  # Default large cost
        
        cost_after = None
        
        # Simulate user response (β ∈ {0,1})
        user_response = np.random.binomial(1, self.acceptance_rate)
        
        if user_response:
            # User accepts the action - modify the request
            self.route.modify(delta_o, delta_d, self.current_request_index)
            self.M = self.route.M  # Get updated M
            
            # `modify` calls `solve()` internally, so cost is fresh
            cost_after = self.route.cost.get('total_cost', float('inf'))
            if np.isnan(cost_after) or np.isinf(cost_after):
                cost_after = 1000.0  # Default large cost
            
            # Calculate reward as cost improvement
            reward = cost_before - cost_after  # positive if we reduced cost
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -100.0, 100.0)
        else:
            reward = -self.alpha
        
        # Move to next request
        self.current_request_index += 1
        self.episode_step += 1
        
        # Check if episode is done
        done = self.current_request_index >= len(self.route.reqs)
        
        info = {
            'user_response': user_response,
            'episode_step': self.episode_step,
            'total_requests': len(self.route.reqs),
            'current_request_index': self.current_request_index,
            'cost_before': cost_before,
            'cost_after': cost_after,
            'accepted': bool(user_response)
        }
        
        return self.M.copy(), self._get_current_request(), reward, done, info
    
    def get_state(self) -> Tuple[np.ndarray, Dict]:
        """
        Get current state
        
        Returns:
        --------
        state: np.ndarray
            Current aggregate tensor
        request: Dict
            Current request
        """
        return self.M.copy(), self._get_current_request()
    
    def render(self):
        """Render current state (for debugging)"""
        print(f"Episode Step: {self.episode_step}")
        print(f"Current Request Index: {self.current_request_index}")
        print(f"Current Request: {self._get_current_request()}")
        print(f"Total Requests: {len(self.route.reqs)}")
        print(f"Aggregate Tensor Shape: {self.M.shape}")
        print(f"Non-zero entries in aggregate tensor: {np.count_nonzero(self.M)}")
        if self.route.cost:
            print(f"Current Cost: {self.route.cost['total_cost']}")


# Example usage and testing
if __name__ == "__main__":
    # Create a Route instance
    route = Route(N=10, auto_solve=False)
    
    # Create environment
    env = DVRPEnvironment(route, acceptance_rate=0.7)
    
    # Reset environment
    state, request = env.reset()
    print("Initial state shape:", state.shape)
    print("Initial request:", request)
    
    # Run a few steps
    for step in range(min(5, len(route.reqs))):
        valid_actions = env.get_valid_actions()
        print(f"\nStep {step + 1}:")
        print(f"Valid actions: {len(valid_actions)}")
        
        if valid_actions:
            # Choose random valid action
            action = valid_actions[np.random.randint(len(valid_actions))]
            print(f"Chosen action: {action}")
            
            # Execute action
            next_state, next_request, reward, done, info = env.step(action)
            print(f"User response: {info['user_response']}")
            print(f"Next request: {next_request}")
            print(f"Reward: {reward}")
            
            if done:
                print("Episode finished!")
                break
        else:
            print("No valid actions available")
            break
    
    # Show final results
    print(f"\nFinal cost: {route.cost}")
    print(f"Final aggregate tensor shape: {env.M.shape}")
    print(f"Non-zero entries: {np.count_nonzero(env.M)}")
