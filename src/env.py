import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy

# Your local modules
from utils import sim, nodify
from utils import solver, user

class RouteEnv(gym.Env):
    """
    Gymnasium environment wrapping the Route object with:
    - Rolling history aggregate of last j fixed requests as a 4-D tensor
    - Next request as a (origin_idx, destination_idx, t_o, t_d) tuple
    - Single pipeline per step: perturb -> user_function accept -> append -> reward (marginal solver delta - patience penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, 
                 N=20, 
                 j=5,                   # number of requests to fix at the beginning
                 lambda_penalty=0.5,
                 max_delta=6,           # max absolute perturbation for time windows (in 30-min ticks)
                 route_kwargs=None,
                 seed: int | None = None,
                 user_function=None,
                 max_solve_time=0.1,
                 mask_action=False): # if True, only valid actions are allowed. Output a 1D boolean mask
        super().__init__()
        self.N = N
        self.j = j
        self.lambda_penalty = float(lambda_penalty)
        self.max_delta = int(max_delta)
        self.max_solve_time = max_solve_time
        
        # Store seed for request generation (deterministic)
        self.request_seed = seed
        
        # User acceptance function (defaults to user.dummy_user)
        self.user_function = user_function if user_function is not None else user.dummy_user

        # Route-like parameters (formerly Route fields), with defaults
        route_kwargs = route_kwargs or {}
        self.vehicle_num = int(route_kwargs.get('vehicle_num', 4))
        self.vehicle_penalty = float(route_kwargs.get('vehicle_penalty', 500.0))
        self.max_vehicles = int(route_kwargs.get('max_vehicles', 10))
        self.vehicle_speed = float(route_kwargs.get('vehicle_speed', 20.0))
        self.depot_node = int(route_kwargs.get('depot_node', 0))
        self.time_window_duration = int(route_kwargs.get('time_window_duration', 30))
        self.vehicle_capacity = int(route_kwargs.get('vehicle_capacity', 4))

        # Seed global random/numpy for deterministic request generation
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

        # Build data formerly produced by Route (now deterministic)
        self.requests_df, self.loc_indices = sim.simulation(N=self.N, vehicle_speed_kmh=self.vehicle_speed)
        
        # Create separate RNG for user responses (will be random but not used for requests)
        self._user_rng = np.random.default_rng(None)  # Always random, not seeded
        self.enc_net = nodify.create_network(self.requests_df)
        self.dm = self.enc_net['distance']
        self.reqs = self.enc_net['requests']

        # Keep an immutable copy of the original sequence (list of dicts)
        self.original_reqs = deepcopy(self.reqs)

        # Fixed set S_i (grows by one each step); we store the *fixed* requests here.
        self.fixed_reqs = []

        # Pointer to the next unprocessed request in original sequence
        self.ptr = None

        # Cached cost of solver for S_{i-1}
        self.cost_prev = None

        # Store masking flag and setup unified action space
        self.mask_action = mask_action
        
        # Unified action space definition: both cases use the same underlying action space
        self.action_dim = (2 * self.max_delta + 1) ** 2  # e.g., 17x17 = 289 actions for max_delta=8
        
        # Create action mapping: action_idx -> (delta_o, delta_d) for both cases
        self._action_map = {}
        idx = 0
        for delta_o in range(-self.max_delta, self.max_delta + 1):
            for delta_d in range(-self.max_delta, self.max_delta + 1):
                self._action_map[idx] = (delta_o, delta_d)
                idx += 1

        self.action_space = spaces.Discrete(self.action_dim)

        # Shapes for observation space
        self.L = len(self.loc_indices)   # number of spatial aggregates (H3 cells)
        self.H = 48                      # time slots (30-min)
        
        # Build observation space
        obs_spaces = {
            "history_aggregate": spaces.Box(low=0, high=np.iinfo(np.int32).max,
                                            shape=(self.L, self.L, self.H, self.H), dtype=np.int32),
            "next_request": spaces.Box(low=np.array([0, 0, 1, 1], dtype=np.float32),
                                       high=np.array([self.L-1, self.L-1, self.H, self.H], dtype=np.float32),
                                       shape=(4,), dtype=np.float32)
        }
        
        # Add action mask to observation space if masking is enabled
        if mask_action:
            obs_spaces["action_mask"] = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.int8)
        
        self.observation_space = spaces.Dict(obs_spaces)

    # ---------- Core Gym API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Note: We don't re-seed for requests since they should stay fixed
        # User responses will remain random via self._user_rng

        # Reset collections
        self.fixed_reqs = []
        self.ptr = 0

        # Seed first j originals as the fixed set
        for k in range(self.j):
            self.fixed_reqs.append(deepcopy(self.original_reqs[self.ptr]))
            self.ptr += 1

        # Cache cost for S_j (before any decisions)
        self.cost_prev = self._solve_cost_for_first_i(len(self.fixed_reqs))

        obs = self._build_observation()
        info = {"i": len(self.fixed_reqs), "accepted": None, "penalty": 0.0}
        return obs, info

    def step(self, action):
        # Both masked and unmasked use the same discrete action space
        action_idx = int(action)
        if action_idx in self._action_map:
            delta_o, delta_d = self._action_map[action_idx]
        else:
            # Fallback to no perturbation if invalid action
            delta_o, delta_d = 0, 0

        # If no more requests, we are done
        done = self.ptr >= self.N
        if done:
            obs = self._build_terminal_observation()
            return obs, 0.0, True, False, {"i": len(self.fixed_reqs), "accepted": None, "penalty": 0.0}

        # Get the next (original) request
        next_req_orig = deepcopy(self.original_reqs[self.ptr])

        # Create candidate perturbed request
        req_pert = self._perturb_request(next_req_orig, delta_o, delta_d)

        # Acceptance via user_function callback
        accepted = bool(self._call_user(next_req_orig, req_pert, delta_o, delta_d))

        # Always append: perturbed if accepted, original if rejected
        self.fixed_reqs.append(req_pert if accepted else next_req_orig)

        # Compute reward: (cost_prev - cost_new) - patience_penalty_if_accepted
        i = len(self.fixed_reqs)

        penalty = self.lambda_penalty * (abs(delta_o) + abs(delta_d)) if accepted else 0.0

        cost_new = self._solve_cost_for_first_i(i)

        # If cost_new is infeasible (inf), set reward to -1000
        if cost_new == float('inf'):
            reward = -500.0
            obs = self._build_terminal_observation()
            info = {"i": i, "accepted": accepted, "penalty": penalty, "marginal_gain": float('inf'),
                    "cost_prev": float(self.cost_prev), "infeasible": True}
            return obs, reward, True, False, info
        
        marginal_gain = (self.cost_prev - cost_new)  # positive if cost decreased

        reward = float(marginal_gain - penalty)

        # Update cache and pointer
        self.cost_prev = cost_new
        self.ptr += 1

        # Build next observation
        obs = self._build_observation()
        done = (self.ptr >= self.N)
        info = {"i": i, "accepted": accepted, "penalty": penalty, "marginal_gain": float(marginal_gain),
                "cost_prev": float(self.cost_prev)}

        return obs, reward, done, False, info

    # ---------- Helpers: Observation / Perturb / Aggregate ----------

    def _build_observation(self):
        """
        Observation dict:
          - history_aggregate: 4-D tensor aggregated from the last j fixed requests
          - next_request: the next unprocessed request (or sentinel zeros if none)
          - action_mask: (optional) 1-D boolean mask for valid actions if mask_action=True
        """
        M = self._aggregate_from_requests(self.fixed_reqs[-self.j:]) if len(self.fixed_reqs) > 0 else np.zeros(
            (self.L, self.L, self.H, self.H), dtype=np.int32
        )

        if self.ptr < self.N:
            nr = self.original_reqs[self.ptr]
            next_tuple = np.array([nr['origin'], nr['destination'], nr['o_t_index'], nr['d_t_index']], dtype=np.float32)
        else:
            # sentinel when done
            next_tuple = np.array([0, 0, 1, 1], dtype=np.float32)

        obs = {"history_aggregate": M, "next_request": next_tuple}
        
        # Add action mask if masking is enabled
        if self.mask_action:
            obs["action_mask"] = self.valid_action_mask().astype(np.int8)
            
        return obs

    def _build_terminal_observation(self):
        # When done, we can still surface the final aggregate and a sentinel next_request.
        return self._build_observation()

    def _call_user(self, next_req_orig, req_pert, delta_o, delta_d):
        """
        Call user function to get acceptance decision.
        Uses the environment's dedicated RNG to ensure user responses are truly random.
        """
        try:
            if self.user_function is not None:
                # Call the user function with the environment's random generator
                # This ensures user responses are random even when requests are seeded
                return bool(self.user_function(self._user_rng))
        except Exception:
            # If the user function errors, fall back to random behavior
            pass

        # Fallback: Default 50% acceptance using dedicated RNG
        acceptance_rate = 0.5
        return bool(self._user_rng.random() <= acceptance_rate)

    def _perturb_request(self, req, delta_o, delta_d):
        """
        Returns a new request dict with modulated time windows, clamped and feasible.
        """
        new_req = deepcopy(req)
        new_req['o_t_index'] = new_req['o_t_index'] + delta_o
        new_req['d_t_index'] = new_req['d_t_index'] + delta_d

        return new_req

    def _aggregate_from_requests(self, reqs_subset):
        """
        Build the 4-D aggregate tensor from a specific list of requests (not from route.reqs).
        Mirrors Route.aggregation() but uses the provided subset.
        """
        M = np.zeros((self.L, self.L, self.H, self.H), dtype=np.int32)
        for r in reqs_subset:
            o_region = int(r['origin'])
            d_region = int(r['destination'])
            o_t = int(np.clip(r['o_t_index'] - 1, 0, self.H - 1))
            d_t = int(np.clip(r['d_t_index'] - 1, 0, self.H - 1))
            # Bounds guard for safety
            if 0 <= o_region < self.L and 0 <= d_region < self.L:
                M[o_region, d_region, o_t, d_t] += 1
        return M

    # ---------- Helpers: Solver Delta on First-i Requests with Distance Subset ----------

    def _solve_cost_for_first_i(self, i):
        """
        Compute total cost for the first i fixed requests via solver.cost_estimator, using:
          - a *subset* distance matrix (rows/cols only for the used locations + depot)
          - a *remapped* requests list so origin/destination indices align with the subset matrix
        This follows your spec:
          - Take top-i requests; incorporate perturbed time windows (already in self.fixed_reqs)
          - Convert origin/dest aggregate index to spatial H3 via encnet['map'] if needed (not strictly required
            here since indices are already integer positions that match rows/cols of the full matrix)
          - Select corresponding rows/cols in the distance matrix to form the subset
          - Remap request indices to local [0..K-1] for the subset matrix
        """
        # 1) First-i requests
        reqs_i = deepcopy(self.fixed_reqs[:i])

        # 2) Collect unique spatial indices used by these requests + ensure depot node is included
        used = set([self.depot_node])
        for r in reqs_i:
            used.add(int(r['origin']))
            used.add(int(r['destination']))
        used = sorted(list(used))

        # 3) Build a mapping old_idx -> new local idx
        old_to_new = {old: new for new, old in enumerate(used)}

        # 4) Subset the distance matrix and remap request indices
        dm_full = self.dm
        dm_sub = dm_full[np.ix_(used, used)]

        # 5) Remap requests to local indices
        remapped_reqs = []
        for r in reqs_i:
            remapped_reqs.append({
                'origin': old_to_new[int(r['origin'])],
                'destination': old_to_new[int(r['destination'])],
                'o_t_index': int(r['o_t_index']),
                'd_t_index': int(r['d_t_index']),
            })

        # 6) Call cost_estimator with the subset matrix and first-i (remapped) requests.
        cost = solver.cost_estimator(
            distance_matrix=dm_sub,
            requests=remapped_reqs,
            vehicle_num=self.vehicle_num,
            vehicle_penalty=self.vehicle_penalty,
            max_vehicles=self.max_vehicles,
            vehicle_travel_speed=self.vehicle_speed,
            time_window_duration=self.time_window_duration,
            vehicle_capacity=self.vehicle_capacity,
            depot_node=old_to_new[self.depot_node],
            max_solve_time=self.max_solve_time
        )

        return float(cost['total_cost'])

    # ---------- Optional: seeding for reproducibility ----------

    def seed(self, seed=None):
        # Seed global random/numpy for request generation consistency
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
        # Keep user responses random (don't seed self._user_rng)
        return [seed]
    
    def valid_action_mask(self):
        """
        Generate action mask for the current state.
        Only works when mask_action=True (discrete action space).
        
        Returns:
            A 1D boolean mask for the action space where True means the action is valid.
        """
        if not self.mask_action:
            raise ValueError("Action masking only available when mask_action=True")
            
        if self.ptr >= self.N:
            # No more requests, all actions invalid (episode should be done)
            return np.zeros(self.action_dim, dtype=bool)
            
        # Get current request
        current_req = self.original_reqs[self.ptr]
        o_t = current_req['o_t_index']
        d_t = current_req['d_t_index']
        
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for action_idx in range(self.action_dim):
            delta_o, delta_d = self._action_map[action_idx]
            
            # Check if the perturbed times are valid
            new_o_t = o_t + delta_o
            new_d_t = d_t + delta_d
            
            # Valid if:
            # 1. Both times are within [1, H] (as per your clipping logic)
            # 2. Pickup time <= dropoff time after perturbation (enforced in _perturb_request)
            if (1 <= new_o_t <= self.H and 1 <= new_d_t <= self.H and new_o_t < new_d_t):
                mask[action_idx] = True
                
        return mask

if __name__ == "__main__":
    env = RouteEnv(user_function=user.dummy_user, mask_action=True)
    obs, info = env.reset()
    #print(obs)
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation)
    print(reward)
    print(info)

    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    print(info)