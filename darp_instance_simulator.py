import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_darp_instance(n_requests=20, coord_range=100, min_pickup_time=300, max_pickup_time=1000, pickup_time_window=30, velocity=1):

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

            # Euclidean distance as travel time (velocity=1)
            distance = np.sqrt((d1 - p1)**2 + (d2 - p2)**2)

            dropoff_start = pickup_start + 2 * distance  # ensure some buffer
            dropoff_end = dropoff_start + pickup_time_window

            time_windows.append(((pickup_start, pickup_end), (dropoff_start, dropoff_end)))
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

if __name__ == "__main__":
    instance = generate_darp_instance()
    print(instance)