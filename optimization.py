import copy
from scipy.spatial import KDTree
import numpy as np
import itertools
import math


def calculate_distance_(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """ Calculate the distance between two coordinates (Haversine) in km.    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Raggio della Terra in km    
    return c * r

def calculate_triangle_distance(tp1, tp2, tp3):
    """Calculate the distance of the triangle formed by the three turnpoints."""

    leg1 = calculate_distance_(tp1.lat, tp1.lon, tp2.lat, tp2.lon)
    leg2 = calculate_distance_(tp2.lat, tp2.lon, tp3.lat, tp3.lon)
    leg3 = calculate_distance_(tp3.lat, tp3.lon, tp1.lat, tp1.lon)
    
    return leg1 + leg2 + leg3, leg1, leg2, leg3

def calculate_total_distance(start_point, tp1, tp2, tp3, finish_point):
    """  
    Calculate the total distance of the complete route including start and finish.  
    """
    start_to_tp1 = 0 # calculate_distance_(start_point.lat, start_point.lon, tp1.lat, tp1.lon)
    triangle_distance, leg1, leg2, leg3 = calculate_triangle_distance(tp1, tp2, tp3)
    tp3_to_finish = 0# calculate_distance_(tp3.lat, tp3.lon, finish_point.lat, finish_point.lon)
    
    total_distance = triangle_distance
    return total_distance, triangle_distance, start_to_tp1, leg1, leg2, leg3, tp3_to_finish



def convert_to_dict(point):
    """Converte un oggetto TrackPoint in un dizionario se necessario."""
    if hasattr(point, 'lat') and hasattr(point, 'lon'):
        # È un oggetto TrackPoint
        return {
            'lat': point.lat,
            'lon': point.lon,
            'time': point.time if hasattr(point, 'time') else None,
            'alt': point.alt if hasattr(point, 'alt') else None
        }
    else:
        # È già un dizionario
        return point


def optimize_track(tracklog, best_score_info):    
    if best_score_info.get('type') == 'triangle':
        optimized_info = optimize_track_triangle(tracklog, best_score_info, 100)
    else:
        optimized_info = optimize_track_line(tracklog, best_score_info, 100)
    
    return optimized_info

def optimize_track_triangle(tracklog, best_score_info, search_radius=1):
    """
    Optimize the route by searching for the best combination of turnpoints (TP1, TP2, TP3)
    within the defined neighborhood, maximizing the total distance with a constraint on the shortest leg.

    :param tracklog: List of track points
    :param best_score_info: Information about the current best score
    :param search_radius: Number of points to consider in both directions
    :return: Information about the optimized best score
    """

    optimized_info = copy.deepcopy(best_score_info)
    
    if 'FAI' in best_score_info.get('triangle_type', ''):
        ratio = 0.28
    else:
        ratio = 0
    
    
    start_point = tracklog[best_score_info['turnpoints_index'][0]]
    finish_point = tracklog[best_score_info['turnpoints_index'][4]]
    closing_distance = best_score_info['properties']['closing_distance']
    
    tp_indices = best_score_info['turnpoints_index']
    tp1_idx, tp2_idx, tp3_idx = tp_indices[1], tp_indices[2], tp_indices[3]
    
    # Intorni per TP1, TP2, TP3
    range_tp1 = range(max(0, tp1_idx - search_radius), min(len(tracklog), tp1_idx + search_radius + 1))
    range_tp2 = range(max(0, tp2_idx - search_radius), min(len(tracklog), tp2_idx + search_radius + 1))
    range_tp3 = range(max(0, tp3_idx - search_radius), min(len(tracklog), tp3_idx + search_radius + 1))
    
    best_total_distance = 0
    best_combo = (tp1_idx, tp2_idx, tp3_idx)
    best_legs = ()
    
    for i1, i2, i3 in itertools.product(range_tp1, range_tp2, range_tp3):
        # Verifica che i punti siano in ordine temporale
        if not (i1 < i2 < i3):
            continue
        
        tp1 = tracklog[i1]
        tp2 = tracklog[i2]
        tp3 = tracklog[i3]
        
        total_distance, triangle_distance, start_to_tp1, leg1, leg2, leg3, tp3_to_finish = calculate_total_distance(
            start_point, tp1, tp2, tp3, finish_point
        )
        
        total_distance = total_distance - closing_distance
        shortest_leg = min(leg1, leg2, leg3)
        
        if total_distance > best_total_distance and shortest_leg >= ratio * total_distance:
            best_total_distance = total_distance
            best_combo = (i1, i2, i3)
            best_legs = (leg1, leg2, leg3, triangle_distance, start_to_tp1, tp3_to_finish)
            #print (total_distance)
    
    

    # Update optimized_info with the best combination found
    i1, i2, i3 = best_combo
    tp1, tp2, tp3 = tracklog[i1], tracklog[i2], tracklog[i3]
    leg1, leg2, leg3, triangle_distance, start_to_tp1, tp3_to_finish = best_legs
    
    optimized_info['turnpoints_index'][1] = i1
    optimized_info['turnpoints_index'][2] = i2
    optimized_info['turnpoints_index'][3] = i3
    
    optimized_info['turnpoints_data'] = [
        convert_to_dict(tp1),
        convert_to_dict(tp2),
        convert_to_dict(tp3)
    ]
    
    optimized_info['properties']['leg1'] = leg1
    optimized_info['properties']['leg2'] = leg2
    optimized_info['properties']['leg3'] = leg3
    optimized_info['properties']['total_distance'] = best_total_distance
    optimized_info['properties']['shortest_leg'] = min(leg1, leg2, leg3)
    optimized_info['properties']['shortest_leg_ratio'] = min(leg1, leg2, leg3) / triangle_distance if triangle_distance > 0 else 0
    optimized_info['properties']['start_to_tp1'] = start_to_tp1
    optimized_info['properties']['tp3_to_finish'] = tp3_to_finish
    optimized_info['properties']['tp1'] = convert_to_dict(tp1)
    optimized_info['properties']['tp2'] = convert_to_dict(tp2)
    optimized_info['properties']['tp3'] = convert_to_dict(tp3)
    
    original_multiplier = best_score_info['properties']['multiplier']
    optimized_info['score'] = best_total_distance * original_multiplier

    # plot_track_and_points(tracklog, optimized_info)
    
    return optimized_info

def optimize_track_line(tracklog, best_score_info, search_radius=1):
    optimized_info = copy.deepcopy(best_score_info)

    start_index = best_score_info['turnpoints_index'][0]
    finish_index = best_score_info['turnpoints_index'][4]
    start_point = tracklog[start_index]
    finish_point = tracklog[finish_index]

    tp_indices = best_score_info['turnpoints_index']
    tp1_idx, tp2_idx, tp3_idx = tp_indices[1], tp_indices[2], tp_indices[3]
    

    range_tp1 = range(max(0, tp1_idx - search_radius), min(len(tracklog), tp1_idx + search_radius + 1))
    range_tp2 = range(max(0, tp2_idx - search_radius), min(len(tracklog), tp2_idx + search_radius + 1))
    range_tp3 = range(max(0, tp3_idx - search_radius), min(len(tracklog), tp3_idx + search_radius + 1))
    
    best_total_distance = 0
    best_combo = (tp1_idx, tp2_idx, tp3_idx)
    best_legs = ()
    
    for i1, i2, i3 in itertools.product(range_tp1, range_tp2, range_tp3):        
        if not (i1 < i2 < i3):
            continue

        tp1 = tracklog[i1]
        tp2 = tracklog[i2]
        tp3 = tracklog[i3]

        start_to_tp1 = calculate_distance_(start_point.lat, start_point.lon, tp1.lat, tp1.lon)
        leg1 = calculate_distance_(tp1.lat, tp1.lon, tp2.lat, tp2.lon)
        leg2 = calculate_distance_(tp2.lat, tp2.lon, tp3.lat, tp3.lon)
        tp3_to_finish = calculate_distance_(tp3.lat, tp3.lon, finish_point.lat, finish_point.lon)

        

        total_distance = start_to_tp1 + leg1 + leg2 + tp3_to_finish

        if total_distance > best_total_distance:
            best_total_distance = total_distance
            best_combo = (i1, i2, i3)
            best_legs = (start_to_tp1, leg1, leg2, tp3_to_finish)
            

    i1, i2, i3 = best_combo
    tp1, tp2, tp3 = tracklog[i1], tracklog[i2], tracklog[i3]
    start_to_tp1, leg1, leg2, tp3_to_finish = best_legs
    
    # Recalculate start point and end point using the new TP1 and TP3
    start_idx,finish_idx = _find_best_end_for_free_distance_idx(tracklog,i1,i3)   
      
    
    optimized_info['turnpoints_index'][1] = i1
    optimized_info['turnpoints_index'][2] = i2
    optimized_info['turnpoints_index'][3] = i3
    optimized_info['turnpoints_index'][0] = start_idx
    optimized_info['turnpoints_index'][4] = finish_idx
    

    optimized_info['turnpoints_data'] = [
        convert_to_dict(tp1),
        convert_to_dict(tp2),
        convert_to_dict(tp3)
    ]

    optimized_info['properties']['leg1'] = leg1
    optimized_info['properties']['leg2'] = leg2
    optimized_info['properties']['leg3'] = 0  # opzionale
    optimized_info['properties']['start_to_tp1'] = start_to_tp1
    optimized_info['properties']['tp3_to_finish'] = tp3_to_finish
    optimized_info['properties']['shortest_leg'] = min(leg1, leg2)
    optimized_info['properties']['shortest_leg_ratio'] = 0  # opzionale, può essere tolto

    optimized_info['properties']['tp1'] = convert_to_dict(tp1)
    optimized_info['properties']['tp2'] = convert_to_dict(tp2)
    optimized_info['properties']['tp3'] = convert_to_dict(tp3)

    optimized_info['properties']['total_distance'] = best_total_distance

    original_multiplier = best_score_info['properties']['multiplier']
    optimized_info['score'] = best_total_distance * original_multiplier

    #plot_track_and_points(tracklog, optimized_info)

    return optimized_info



def _find_best_end_for_free_distance_idx(tracklog, tp1_idx: int, tp3_idx: int) -> tuple[int, int]:
        """
        Find the indices of:
        1. The point in tracklog[:tp1_idx] that is farthest from tp1
        2. The point in tracklog[tp3_idx:] that is farthest from tp3
        
        Uses spatial indexing with KD-Tree for efficient distance calculations.
        
        Args:
            tp1_idx: Index of the tp1 point
            tp3_idx: Index of the tp3 point
            
        Returns:
            tuple: (idx_farthest_from_tp1, idx_farthest_from_tp3)
        """
        # Extract the reference points
        tp1_point = tracklog[tp1_idx]
        tp3_point = tracklog[tp3_idx]
        
        # Define the subsets
        subset_start = tracklog[:tp1_idx]
        subset_end = tracklog[tp3_idx:]
        
        # Initialize default values in case subsets are empty
        idx_farthest_from_tp1 = 0
        idx_farthest_from_tp3 = tp3_idx
        
        # Process the start subset if not empty
        if subset_start:
            # Convert geographic coordinates to radians
            points_rad = np.radians(np.array([(point.lat, point.lon) for point in subset_start]))
            tp1_rad = np.radians([tp1_point.lat, tp1_point.lon])
            
            # Create KD-tree for the start subset
            tree_start = KDTree(points_rad)
            
            # Find the point with maximum distance (query with largest k and take the last one)
            distances, indices = tree_start.query(tp1_rad, k=len(subset_start))
            
            # The farthest point is the last one in the sorted distances
            idx_farthest_from_tp1 = indices[-1]
        
        # Process the end subset if not empty
        if subset_end:
            # Convert geographic coordinates to radians
            points_rad = np.radians(np.array([(point.lat, point.lon) for point in subset_end]))
            tp3_rad = np.radians([tp3_point.lat, tp3_point.lon])
            
            # Create KD-tree for the end subset
            tree_end = KDTree(points_rad)
            
            # Find the point with maximum distance
            distances, indices = tree_end.query(tp3_rad, k=len(subset_end))
            
            # The farthest point is the last one in the sorted distances
            farthest_relative_idx = indices[-1]
            
            # Convert back to original index
            idx_farthest_from_tp3 = farthest_relative_idx + tp3_idx
        
        return idx_farthest_from_tp1, idx_farthest_from_tp3