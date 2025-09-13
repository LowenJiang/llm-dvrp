import pandas as pd
import numpy as np
import h3
import random
from geopy.distance import geodesic
from scipy.stats import multivariate_normal

"""
This module is used to generate ride requests for DARP simulation. 
The output contains origin, destination, time window of pickup and dropoff.
Origin and destination are in h3 indices. 
time window are indices of 30min intervals
"""

def simulation(N=20, resolution=7, temporal_interval_minutes=30, vehicle_speed_kmh=20, min_distance_h3_units=3, random_seed=None):
    """
    Generate N ride requests in San Francisco City Region for DARP simulation.
    
    This function creates realistic ride-sharing requests by:
    1. Defining San Francisco's geographic boundaries using H3 hexagonal indexing
    2. Generating pickup and dropoff locations from a 2D Gaussian distribution
    3. Ensuring spatial constraints (minimum distance between pickup/dropoff)
    4. Calculating realistic temporal constraints based on travel time and vehicle speed
    5. Assigning time windows within a 24-hour period divided into intervals
    
    Parameters:
    -----------
    N : int, default=20
        Number of ride requests to generate
    resolution : int, default=9
        H3 resolution level (higher = smaller hexagons, more precise)
    temporal_interval_minutes : int, default=30
        Time interval duration in minutes (24h is divided by this)
    vehicle_speed_kmh : float, default=30
        Average vehicle speed in km/h for travel time calculations
    min_distance_h3_units : int, default=3
        Minimum distance between pickup and dropoff in H3 resolution units
    
    Returns:
    --------
    tuple
        (pd.DataFrame, list)
        - DataFrame with columns: origin, destination, o_t_index, d_t_index
          - origin: Origin (pickup) H3 index
          - destination: Destination (dropoff) H3 index  
          - o_t_index: Origin time window index (1-48 for 30min intervals)
          - d_t_index: Destination time window index (1-48 for 30min intervals)
        - List of H3 indices covering the San Francisco region
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # San Francisco inner city boundaries (lat, lon) - more restrictive to stay within city limits
    sf_center_lat, sf_center_lon = 37.7749, -122.4194
    sf_bounds = {
        'north': 37.8100,
        'south': 37.7300, 
        'east': -122.3700,
        'west': -122.5000
    }
    
    # Generate all H3 indices covering San Francisco at given resolution
    sf_h3_indices = []
    
    # Create a grid of points within SF bounds and get their H3 indices
    lat_range = np.linspace(sf_bounds['south'], sf_bounds['north'], 50)
    lon_range = np.linspace(sf_bounds['west'], sf_bounds['east'], 50)
    
    for lat in lat_range:
        for lon in lon_range:
            h3_index = h3.latlng_to_cell(lat, lon, resolution)
            if h3_index not in sf_h3_indices:
                sf_h3_indices.append(h3_index)
    
    # Calculate number of time intervals in 24 hours
    num_time_intervals = int(24 * 60 / temporal_interval_minutes)  # 48 for 30min intervals
    
    # Set up 2D Gaussian distribution centered on SF
    mean = [sf_center_lat, sf_center_lon]
    # Covariance matrix - adjust spread to cover SF inner city area
    cov = [[0.005, 0], [0, 0.005]]  # Smaller spread to stay within inner city bounds
    
    requests = []
    
    for _ in range(N):
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # Sample pickup and dropoff locations from 2D Gaussian
            pickup_lat, pickup_lon = multivariate_normal.rvs(mean, cov)
            dropoff_lat, dropoff_lon = multivariate_normal.rvs(mean, cov)
            
            # Convert to H3 indices
            pickup_h3 = h3.latlng_to_cell(pickup_lat, pickup_lon, resolution)
            dropoff_h3 = h3.latlng_to_cell(dropoff_lat, dropoff_lon, resolution)
            
            # Check if both locations are within SF bounds
            if (pickup_h3 in sf_h3_indices and dropoff_h3 in sf_h3_indices and 
                pickup_h3 != dropoff_h3):
                
                # Check minimum distance constraint
                pickup_center = h3.cell_to_latlng(pickup_h3)
                dropoff_center = h3.cell_to_latlng(dropoff_h3)
                
                # Calculate distance in km
                distance_km = geodesic(pickup_center, dropoff_center).kilometers
                # Calculate H3 unit distance (approximate)
                h3_edge_length_km = h3.average_hexagon_edge_length(resolution, unit='km')
                distance_h3_units = distance_km / h3_edge_length_km
                
                if distance_h3_units >= min_distance_h3_units:
                    # Calculate minimum travel time in intervals
                    travel_time_hours = distance_km / vehicle_speed_kmh * 2

                    travel_time_intervals = int(np.ceil(travel_time_hours * 60 / temporal_interval_minutes)) + 2
                    
                    # Generate pickup time window (1 to num_time_intervals)
                    # Ensure enough time remains for dropoff
                    max_pickup_time = num_time_intervals - travel_time_intervals
                    if max_pickup_time > 0:
                        pickup_time_index = random.randint(10, 40)
                        dropoff_time_index = pickup_time_index + travel_time_intervals
                        
                        # Ensure dropoff time is within valid range
                        if dropoff_time_index <= num_time_intervals:
                            requests.append({
                                'origin': pickup_h3,
                                'destination': dropoff_h3,
                                'o_t_index': pickup_time_index,
                                'd_t_index': dropoff_time_index
                            })
                            break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not generate valid request after {max_attempts} attempts")
    
    return pd.DataFrame(requests), sf_h3_indices


if __name__ == "__main__":
    requests, sf_h3_indices = simulation()
    print(requests)
    print(f"\nH3 indices covering SF region ({len(sf_h3_indices)} total):")
    print(sf_h3_indices)
