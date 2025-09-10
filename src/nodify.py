"""
The goal of this module is to encode a network to be input into the cost solver. 
"""

import pandas as pd
import numpy as np
import h3
from geopy.distance import geodesic

def create_network(requests_df, resolution=9):
    """
    Create a network representation from ride requests.
    
    Parameters:
    -----------
    requests_df : pd.DataFrame
        DataFrame with columns: origin, destination, o_t_index, d_t_index
    resolution : int, default=9
        H3 resolution level for depot location
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - map: dict mapping spatial indices to H3 indices
        - distance: numpy array of distances between locations in km
        - requests: list of requests with H3 indices converted to spatial IDs
        - depot: H3 index of the depot location
    """
    # Calculate depot location (center of SF area)
    sf_bounds = {
        'north': 37.8100,
        'south': 37.7300, 
        'east': -122.3700,
        'west': -122.5000
    }
    depot_lat = (sf_bounds['north'] + sf_bounds['south']) / 2
    depot_lon = (sf_bounds['east'] + sf_bounds['west']) / 2
    depot_h3 = h3.latlng_to_cell(depot_lat, depot_lon, resolution)
    
    # Extract all unique H3 indices from origins and destinations
    all_h3_indices = set()
    all_h3_indices.update(requests_df['origin'].unique())
    all_h3_indices.update(requests_df['destination'].unique())
    
    # Add depot as the first location (index 0)
    all_h3_indices.add(depot_h3)
    all_h3_indices = list(all_h3_indices)
    
    # Ensure depot is at index 0
    if depot_h3 in all_h3_indices:
        all_h3_indices.remove(depot_h3)
        all_h3_indices.insert(0, depot_h3)
    
    # Create mapping from spatial index to H3 index
    spatial_to_h3_map = {i: h3_index for i, h3_index in enumerate(all_h3_indices)}
    
    # Create reverse mapping from H3 index to spatial index
    h3_to_spatial_map = {h3_index: i for i, h3_index in enumerate(all_h3_indices)}
    
    # Create distance matrix
    n_locations = len(all_h3_indices)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    # Calculate distances between all pairs of locations
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                # Get lat/lon coordinates for both H3 indices
                coord_i = h3.cell_to_latlng(all_h3_indices[i])
                coord_j = h3.cell_to_latlng(all_h3_indices[j])
                
                # Calculate geodesic distance in km
                distance_km = geodesic(coord_i, coord_j).kilometers
                distance_matrix[i][j] = distance_km
            else:
                distance_matrix[i][j] = 0.0
    
    # Convert requests from H3 indices to spatial IDs
    requests_with_spatial_ids = []
    for _, row in requests_df.iterrows():
        request = {
            'origin': h3_to_spatial_map[row['origin']],
            'destination': h3_to_spatial_map[row['destination']],
            'o_t_index': row['o_t_index'],
            'd_t_index': row['d_t_index']
        }
        requests_with_spatial_ids.append(request)
    
    return {
        "map": spatial_to_h3_map,
        "distance": distance_matrix,
        "requests": requests_with_spatial_ids,
        "depot": depot_h3
    }
    """
    Create a network representation from ride requests.
    
    Parameters:
    -----------
    requests_df : pd.DataFrame
        DataFrame with columns: origin, destination, o_t_index, d_t_index
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - "map": dict mapping spatial indices to H3 indices
        - "distance": numpy array of distances between locations in km
        - "requests": list of requests with H3 indices converted to spatial IDs
    """
    # Extract all unique H3 indices from origins and destinations
    all_h3_indices = set()
    all_h3_indices.update(requests_df['origin'].unique())
    all_h3_indices.update(requests_df['destination'].unique())
    all_h3_indices = list(all_h3_indices)
    
    # Create mapping from spatial index to H3 index
    spatial_to_h3_map = {i: h3_index for i, h3_index in enumerate(all_h3_indices)}
    
    # Create reverse mapping from H3 index to spatial index
    h3_to_spatial_map = {h3_index: i for i, h3_index in enumerate(all_h3_indices)}
    
    # Create distance matrix
    n_locations = len(all_h3_indices)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    # Calculate distances between all pairs of locations
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                # Get lat/lon coordinates for both H3 indices
                coord_i = h3.cell_to_latlng(all_h3_indices[i])
                coord_j = h3.cell_to_latlng(all_h3_indices[j])
                
                # Calculate geodesic distance in km
                distance_km = geodesic(coord_i, coord_j).kilometers
                distance_matrix[i][j] = distance_km
            else:
                distance_matrix[i][j] = 0.0
    
    # Convert requests from H3 indices to spatial IDs
    requests_with_spatial_ids = []
    for _, row in requests_df.iterrows():
        request = {
            'origin': h3_to_spatial_map[row['origin']],
            'destination': h3_to_spatial_map[row['destination']],
            'o_t_index': row['o_t_index'],
            'd_t_index': row['d_t_index']
        }
        requests_with_spatial_ids.append(request)
    
    return {
        "map": spatial_to_h3_map,
        "distance": distance_matrix,
        "requests": requests_with_spatial_ids
    }
