# DVRP Environment Documentation

## Overview

The DVRP (Dynamic Vehicle Routing Problem) Environment implements the state, action, and transition dynamics as described in the mathematical formulation. This environment simulates a ride-sharing system where the agent can suggest time window adjustments to users, and users may accept or reject these suggestions.

## Environment Structure

### State Space (S = M × R)

The state consists of two components:

1. **Aggregate Tensor (M)**: A 5-dimensional tensor of shape `(ℓ, ℓ, h, h, c_max)` where:
   - `ℓ` = number of spatial regions (default: 50)
   - `h` = number of time intervals (default: 48 for 24h/30min intervals)
   - `c_max` = maximum customer count per request (default: 4)

2. **Current Request (R)**: A dictionary containing:
   - `origin`: origin region index (0 to ℓ-1)
   - `destination`: destination region index (0 to ℓ-1)
   - `o_t_index`: pickup time index (0 to h-1)
   - `d_t_index`: dropoff time index (0 to h-1)
   - `customers`: number of customers (1 to c_max)

### Action Space (A)

Actions are pairs of integer offsets `(δ^o, δ^d)` where:
- `δ^o` ∈ [-h, h]: offset for pickup time
- `δ^d` ∈ [-h, h]: offset for dropoff time

**Feasibility Constraints**:
- `0 ≤ t^o' + δ^o < h`
- `0 ≤ t^d' + δ^d < h`
- `t^o' + δ^o ≤ t^d' + δ^d` (pickup before dropoff)

### Transition Dynamics

1. **User Response**: β ∈ {0,1} drawn from `P(β | s, a)`
   - β = 1: User accepts the action (time shifts applied)
   - β = 0: User rejects the action (original times kept)

2. **State Update**: 
   ```
   M_{i+1} = M_i + e(o', d', t^o' + β·δ^o, t^d' + β·δ^d, c)
   ```
   where `e(·)` is the unit tensor at the specified index.

3. **Next Request**: New request drawn i.i.d. from the distribution.

## Key Features

### H3 Indexing System
- Uses H3 hexagonal indexing for spatial regions
- Converts H3 indices to region indices for tensor operations
- Maintains mapping between H3 indices and region indices

### Request Generation
- Generates realistic ride requests using the existing simulation
- Converts H3 coordinates to region indices
- Ensures temporal feasibility (pickup before dropoff)

### Route Optimization
- Integrates with existing Route class and solver
- Converts episode requests back to H3 format for optimization
- Returns comprehensive solution including cost, routes, and feasibility

## Usage Example

```python
from src.env import DVRPEnvironment

# Create environment
env = DVRPEnvironment(
    N=20,                    # 20 requests per episode
    num_regions=50,          # 50 spatial regions
    num_time_intervals=48,   # 48 time intervals (24h/30min)
    max_customers=4,         # Max 4 customers per request
    acceptance_rate=0.5      # 50% user acceptance rate
)

# Reset environment
state, request = env.reset()

# Run episode
for step in range(env.N):
    # Get valid actions
    valid_actions = env.get_valid_actions()
    
    # Choose action (e.g., random)
    action = valid_actions[np.random.randint(len(valid_actions))]
    
    # Execute action
    next_state, next_request, reward, done, info = env.step(action)
    
    if done:
        break

# Solve route optimization
solution = env.solve_route()
```

## Environment Parameters

### Route Optimization Parameters
- `N`: Number of requests per episode
- `vehicle_num`: Number of vehicles
- `vehicle_penalty`: Penalty for using additional vehicles
- `max_vehicles`: Maximum number of vehicles
- `vehicle_speed`: Vehicle speed in km/h
- `depot_node`: Depot node index
- `time_window_duration`: Time window duration in minutes
- `vehicle_capacity`: Vehicle capacity
- `max_solve_time`: Maximum solve time in seconds

### Environment-Specific Parameters
- `num_regions`: Number of spatial regions (ℓ)
- `num_time_intervals`: Number of time intervals (h)
- `max_customers`: Maximum customer count per request (c_max)
- `acceptance_rate`: User acceptance rate for actions

## Methods

### Core Methods
- `reset()`: Reset environment and return initial state
- `step(action)`: Execute action and return next state
- `get_valid_actions()`: Get list of valid actions for current request
- `get_state()`: Get current state
- `solve_route()`: Solve route optimization for current episode

### Utility Methods
- `render()`: Print current state information
- `_generate_request()`: Generate new request
- `_h3_to_region_index()`: Convert H3 index to region index

## Output Format

### State
- `state`: 5D numpy array of shape (ℓ, ℓ, h, h, c_max)
- `request`: Dictionary with request information

### Action
- `action`: Tuple of (δ^o, δ^d) offsets

### Transition
- `next_state`: Updated 5D numpy array
- `next_request`: Next request dictionary
- `reward`: Reward value (placeholder for now)
- `done`: Boolean indicating episode completion
- `info`: Dictionary with additional information

### Route Solution
- `status`: 'FEASIBLE' or 'INFEASIBLE'
- `total_distance_km`: Total routing distance
- `num_vehicles_used`: Number of vehicles used
- `routes`: List of vehicle routes
- `routing_cost`: Routing cost
- `vehicle_penalty_cost`: Penalty for additional vehicles
- `total_cost`: Total cost (routing + penalty)
- `solve_time`: Time taken to solve

## Mathematical Formulation

The environment implements the following mathematical model:

**State**: s_i = ⟨M_i, ρ_i⟩ where M_i is the aggregate tensor and ρ_i is the current request.

**Action**: a_i = (δ^o, δ^d) with feasibility constraints.

**Transition**: 
```
P((M', ρ') | (M_i, ρ_i), a_i) = Σ_{β∈{0,1}} P(β | s_i, a_i) · 1{M' = M_{i+1}^{(β)}} · P_D(ρ')
```

**Update Rule**:
```
M_{i+1}^{(β)} = M_i + e(o', d', t^o' + β·δ^o, t^d' + β·δ^d, c)
```

This implementation provides a complete foundation for reinforcement learning experiments with the DVRP problem.
