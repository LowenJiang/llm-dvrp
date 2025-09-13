# env_route.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
import random
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
                 lambda_penalty=0.1,
                 max_delta=6,           # max absolute perturbation for time windows (in 30-min ticks)
                 route_kwargs=None,
                 seed: int | None = None,
                 user_function=None,
                 random_seed=None):
        super().__init__()

        self.N = N
        self.j = j
        self.lambda_penalty = float(lambda_penalty)
        self.max_delta = int(max_delta)
        self._rng = np.random.default_rng(seed)

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
        self.max_solve_time = float(route_kwargs.get('max_solve_time', 5.0))

        # Build data formerly produced by Route
        self.requests_df, self.loc_indices = sim.simulation(N=self.N, vehicle_speed_kmh=self.vehicle_speed)
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

        # Shapes for observation space
        self.L = len(self.loc_indices)   # number of spatial aggregates (H3 cells)
        self.H = 48                      # time slots (30-min)
        # Observation = {"history_aggregate": (L,L,H,H) int tensor, "next_request": (4,) float/int tuple}
        self.observation_space = spaces.Dict({
            "history_aggregate": spaces.Box(low=0, high=np.iinfo(np.int32).max,
                                            shape=(self.L, self.L, self.H, self.H), dtype=np.int32),
            "next_request": spaces.Box(low=np.array([0, 0, 1, 1], dtype=np.float32),
                                       high=np.array([self.L-1, self.L-1, self.H, self.H], dtype=np.float32),
                                       shape=(4,), dtype=np.float32)
        })

        # Action = (delta_o_t, delta_d_t) as integer perturbations directly in [-max_delta, +max_delta]
        self.action_space = spaces.Box(low=-self.max_delta, high=self.max_delta, shape=(2,), dtype=np.float32)

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # ---------- Core Gym API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

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
    
        """
        Info contains the following information:
        - i: number of requests fixed
        - accepted: whether the request was accepted
        - penalty: patience penalty if accepted
        - marginal_gain: marginal gain from the solver (reward - penalty). Almost reward is negative.
        - cost_prev: cost of the previous requests
        - infeasible: whether the routing became infeasible
        """

    def step(self, action):
        # Handle continuous actions
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -self.max_delta, self.max_delta)
        delta_o = int(np.round(a[0]))  # Round to nearest integer
        delta_d = int(np.round(a[1]))

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
        cost_new = self._solve_cost_for_first_i(i)
        
        # Check if routing became infeasible - end episode early
        if np.isinf(cost_new):
            obs = self._build_terminal_observation()
            return obs, -1000.0, True, False, {"i": i, "accepted": accepted, "penalty": penalty, 
                                               "marginal_gain": 0.0, "cost_prev": float(self.cost_prev), 
                                               "infeasible": True}
        
        marginal_gain = (self.cost_prev - cost_new)  # positive if cost decreased
        penalty = self.lambda_penalty * (abs(delta_o) + abs(delta_d)) if accepted else 0.0
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

        return {"history_aggregate": M, "next_request": next_tuple}

    def _build_terminal_observation(self):
        # When done, we can still surface the final aggregate and a sentinel next_request.
        return self._build_observation()

    def _call_user(self, next_req_orig, req_pert, delta_o, delta_d):
        """
        Call the configured user_function with flexible signatures.
        Tries (orig, pert, delta_o, delta_d) -> bool, then () -> bool, then (pert) -> bool.
        """
        try:
            return self.user_function(next_req_orig, req_pert, delta_o, delta_d)
        except TypeError:
            try:
                return self.user_function()
            except TypeError:
                return self.user_function(req_pert)

    # Wait.. this action space need some more scrutiny. Where should we put the constraints? -> logits in policy?
    def _perturb_request(self, req, delta_o, delta_d):

        """
        Returns a new request dict with modulated time windows, clamped and feasible.
        """

        """
        ### In actual action, we hope to handle masking in policy.
        # enforce pickup before dropoff
        if new_req['o_t_index'] >= new_req['d_t_index']:
            new_req['d_t_index'] = min(self.H, new_req['o_t_index'] + 1)
        """

        new_req = deepcopy(req)
        new_req['o_t_index'] = int(np.clip(new_req['o_t_index'] + delta_o, 1, self.H))
        new_req['d_t_index'] = int(np.clip(new_req['d_t_index'] + delta_d, 1, self.H))
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

        # Check if routing is infeasible
        if cost['status'] == 'INFEASIBLE':
            return float('inf')  # Special value to indicate infeasible routing
        
        return float(cost['total_cost'])

    # ---------- Optional: seeding for reproducibility ----------

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed]

if __name__ == "__main__":
    env = RouteEnv(user_function=user.dummy_user, random_seed=42)
    obs, info = env.reset()
    #print(obs)
    #print(info)
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation)
    print(reward)
    print(info)
