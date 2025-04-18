import numpy as np
import itertools
from typing import List, Dict, Optional
import datetime
import math
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
import logging
import time
import optimization


logger = logging.getLogger(__name__)


class XCScorer:
    def __init__(self, tracklog_or_df, scoring_rules: Dict):
        """
        Initialize the XC scoring calculator.

        Args:
            tracklog_or_df: Either a List of trackpoints with lat, lon, and time attributes,
                           or a pandas DataFrame with lat, lon, time columns
            scoring_rules: Dictionary with multipliers for different flight types
        """
        # Check if input is a DataFrame
        if hasattr(tracklog_or_df, 'iterrows'):
            # Convert DataFrame to tracklog list
            self.tracklog = self._convert_df_to_tracklog(tracklog_or_df)
        else:
            # Input is already a tracklog list
            self.tracklog = tracklog_or_df

        self.scoring_rules = scoring_rules

        self.take_off = self.tracklog[0]

        # Detect the actual takeoff point
        takeoff_idx = self._detect_takeoff_point()
        # self.start_point_idx = takeoff_idx
        self.first_point = self.tracklog[0]
        self.last_point = self.tracklog[-1]
        self.take_off = self.tracklog[takeoff_idx]
        self.hull_tracklog_indices = None
        # The track is divided into multiple segments depending on its curvature.A straight line results in many segments, while a closed flight forms a single segment.This makes the hull calculation more precise.
        number_of_track_segments = self.compute_curvature_simple()
        self.hull_go(number_of_track_segments)

    def compute_curvature_simple(self):
        """Estimates the curvature of the track and returns a value from 1 (high curvature) to 10 (almost straight)."""
        points = np.array([[p.lat, p.lon] for p in self.tracklog])

        if len(points) < 3:
            return 0  # Not enough points to evaluate curvature

        total_distance = np.sum(np.linalg.norm(
            np.diff(points, axis=0), axis=1))  # Sum of segment lengths
        start_to_end_distance = np.linalg.norm(
            points[-1] - points[0])  # Direct distance from start to end

        if start_to_end_distance == 0:
            return 1  # Maximum curvature (completely looped track)

        ratio = total_distance / start_to_end_distance  # Curvature indicator

        # Map ratio to a scale from 1 to 10
        if ratio <= 1:
            return 10  # Almost straight
        elif ratio >= 4:
            return 1  # Very curved
        else:
            # Linearly interpolate intermediate values
            return round(10 - (ratio - 1) * 3)

    def analyze_track(self):
        """Prints the curvature level and runs the ConvexHull process accordingly."""
        curvature_level = self.compute_curvature_simple()
        print(f"Curvature Level: {curvature_level}")
        self.hull_go(curvature_level)

    def hull_go(self, num_subsets=5):  # Run ConvexHull only once is enough.
        """
        Calculate the convex hull of the tracklog points.
        """
        points = np.array([[p.lat, p.lon] for p in self.tracklog])
        total_points = len(points)
        # truck is divided by num_subsets to get more point
        subset_size = max(10, total_points // num_subsets)

        hull_indices = set()

        for i in range(0, total_points, subset_size):
            subset = points[i:i + subset_size]
            if len(subset) > 2:
                hull = ConvexHull(subset, qhull_options="QJ")
                for idx in hull.vertices:
                    hull_indices.add(i + idx)

        self.hull_tracklog_indices = sorted(hull_indices)

    def _convert_df_to_tracklog(self, df):
        """
        Convert a DataFrame to a tracklog list.

        Args:
            df: DataFrame with columns for time, lat, lon, and altitude

        Returns:
            List of trackpoint objects
        """
        tracklog = []

        for _, row in df.iterrows():
            point = type('TrackPoint', (), {})()
            point.lat = row['lat']
            point.lon = row['lon']
            point.time = row['datetime'] if 'datetime' in row else row['time']
            point.alt = row.get('gps_alt', row.get('pressure_altitude', 0))

            # Add to_dict method to point object
            def to_dict_method(self):
                return {
                    'lat': self.lat,
                    'lon': self.lon,
                    'time': self.time.strftime('%H:%M:%S') if hasattr(self.time, 'strftime') else str(self.time),
                    'alt': self.alt
                }
            point.to_dict = to_dict_method.__get__(point)

            tracklog.append(point)

        return tracklog

    def score_flight(self, track_optimization=False) -> Dict:
        """
        Score the flight using the best possible type (triangle or free distance).
        Returns the scoring information for the best flight type, including max distance.
        """
        process_start_time = datetime.datetime.now()
        logger.info(f"Start processing at {process_start_time}")

        # DISABLE_OPTIMIZATION = not track_optimization

        # Disable out and return scoring for now
        DISABLE_OUT_RETURN = False

        # Score all triangle types in a single pass
        triangle_scores = self.score_all_triangle_types()

        # Score as free distance flight
        free_score = self.score_free_distance_flight(self.scoring_rules)

        # Initialize out_return_score to None
        out_return_score = None

        # Only calculate out_return_score if not disabled
        if not DISABLE_OUT_RETURN:
            # Score as out and return flight
            out_return_score = self.score_out_return_flight(self.scoring_rules)

        # Calculate max distance between any two points
        max_distance_info = self.calculate_max_distance()

        # Combine all scores
        all_scores = triangle_scores + [free_score]

        best_score_info = max(all_scores, key=lambda x: x['score'])

        # Add max distance information to the result
        best_score_info['max_distance_info'] = max_distance_info

        # Only add out_return to the result if it was calculated
        if out_return_score is not None:
            best_score_info['out_return'] = out_return_score

        process_end_time = datetime.datetime.now()
        logger.info(
            f"End processing at {process_end_time}, total time: {process_end_time - process_start_time}")

        if track_optimization:
            best_score_info = optimization.optimize_track(
                self.tracklog, best_score_info)

        return best_score_info

    def _detect_takeoff_point(self):
        """
        Detect the actual takeoff point for a paraglider by analyzing speed,
        altitude changes, and consistent flying patterns directly throughout the tracklog.

        Returns:
            int: Index of the detected takeoff point in the tracklog
        """
        if len(self.tracklog) < 10:
            return 0  # Not enough points to analyze

        speeds = []
        height_speed_products = []
        moving_flags = []

        MIN_TAKEOFF_SPEED = 10.0       # km/h
        MIN_SPEED_INCREASE = 7.0       # km/h increase
        TAKEOFF_SPEED_RATIO = 2.0      # Speed multiplier during takeoff
        MIN_HEIGHT_SPEED_PRODUCT = 20.0
        CONSECUTIVE_POINTS = 5         # Number of points to confirm consistent flight

        # Calculate speed, altitude changes, and height-speed products
        for i in range(1, len(self.tracklog)):
            prev, curr = self.tracklog[i - 1], self.tracklog[i]

            distance = self.calculate_distance(
                prev.lat, prev.lon, curr.lat, curr.lon)

            # Handle different time object types (datetime.time vs. full datetime objects)
            # This works both for IGC parser output and DataFrame input
            if hasattr(curr.time, 'total_seconds') and hasattr(prev.time, 'total_seconds'):
                # These are full datetime objects, can calculate difference directly
                time_diff = (curr.time - prev.time).total_seconds()
            else:
                # These are time objects without date, use current date as reference
                fake_date = datetime.date.today()
                prev_dt = datetime.datetime.combine(fake_date, prev.time)
                curr_dt = datetime.datetime.combine(fake_date, curr.time)
                time_diff = (curr_dt - prev_dt).total_seconds()

                # Handle day change (when a track spans midnight)
                if time_diff < 0:
                    # Add 24 hours to correct for day change
                    time_diff += 86400

            time_diff = max(time_diff, 1)  # Avoid division by zero

            speed = (distance / time_diff) * 3600  # km/h
            alt_diff = curr.alt - prev.alt
            height_speed_product = speed * abs(alt_diff)

            speeds.append(speed)
            height_speed_products.append(height_speed_product)
            moving_flags.append(speed > 5.0)

        # Apply exponential moving average smoothing
        alpha = 0.3  # Adjust alpha for more or less smoothing
        smoothed_speeds = [speeds[0]]
        for speed in speeds[1:]:
            new_speed = alpha * speed + (1 - alpha) * smoothed_speeds[-1]
            smoothed_speeds.append(new_speed)

        # Use smoothed speeds for detection
        speeds = smoothed_speeds

        # Detect clear takeoff transition with debugging info
        for i in range(5, len(speeds) - CONSECUTIVE_POINTS):
            pre_speeds = speeds[i - 5:i]
            post_speeds = speeds[i:i + CONSECUTIVE_POINTS]
            post_hs_products = height_speed_products[i:i + CONSECUTIVE_POINTS]

            avg_pre_speed = sum(pre_speeds) / len(pre_speeds)
            avg_post_speed = sum(post_speeds) / len(post_speeds)
            avg_post_hs_product = sum(post_hs_products) / len(post_hs_products)

            speed_increase = avg_post_speed - avg_pre_speed
            speed_ratio = avg_post_speed / max(avg_pre_speed, 0.5)

            # Debugging output
            # logger.info(f"Index {i}: pre_speed={avg_pre_speed:.2f}, post_speed={avg_post_speed:.2f}/{MIN_TAKEOFF_SPEED}, speed_increase={speed_increase:.2f}/{MIN_SPEED_INCREASE}, speed_ratio={speed_ratio:.2f}/{TAKEOFF_SPEED_RATIO}, hs_product={avg_post_hs_product:.2f}/{MIN_HEIGHT_SPEED_PRODUCT}, moving={all(moving_flags[i:i + CONSECUTIVE_POINTS])}")

            if (
                speed_increase >= MIN_SPEED_INCREASE
                and avg_post_speed >= MIN_TAKEOFF_SPEED
                and (speed_ratio >= TAKEOFF_SPEED_RATIO
                     or avg_post_hs_product >= MIN_HEIGHT_SPEED_PRODUCT)
                and all(moving_flags[i:i + CONSECUTIVE_POINTS])
            ):
                logger.info(f"Takeoff detected at index {i + 1}")
                return i + CONSECUTIVE_POINTS

        # Fallback detection based on sustained flying pattern with debugging info
        for i in range(len(speeds) - CONSECUTIVE_POINTS):
            window_speeds = speeds[i:i + CONSECUTIVE_POINTS]
            window_hs_products = height_speed_products[i:i +
                                                       CONSECUTIVE_POINTS]

            avg_window_speed = sum(window_speeds) / CONSECUTIVE_POINTS
            avg_window_hs_product = sum(
                window_hs_products) / CONSECUTIVE_POINTS

            logger.info(
                f"Fallback check at index {i}: window_speed={avg_window_speed:.2f}, hs_product={avg_window_hs_product:.2f}")

            if (
                all(s >= MIN_TAKEOFF_SPEED for s in window_speeds)
                and avg_window_hs_product >= MIN_HEIGHT_SPEED_PRODUCT
            ):
                logger.info(f"Fallback takeoff detected at index {i + 1}")
                return i + 1

        logger.info(
            "Paraglider takeoff not detected, defaulting to first point")
        return 0

    def calculate_max_distance(self) -> Dict:
        """
        Calculate the maximum distance between any two points in the tracklog
        and identify those points.

        Returns:
            Dictionary with maximum distance (km) and the indices and data of the two points
        """
        if len(self.tracklog) < 2:
            return {
                'max_distance': 0,
                'point1_idx': 0,
                'point2_idx': 0,
                'point1': self.tracklog[0].to_dict() if self.tracklog else None,
                'point2': self.tracklog[0].to_dict() if self.tracklog else None
            }

        # Use vectorized calculation for efficiency with larger tracklogs
        max_distance = 0
        point1_idx = 0
        point2_idx = 0

        # For larger tracklogs, consider using a sampling approach
        if len(self.tracklog) > 1000:
            # Sample points at regular intervals
            step = max(1, len(self.tracklog) // 500)
            sampled_indices = range(0, len(self.tracklog), step)
            tracklog_subset = [self.tracklog[i] for i in sampled_indices]

            # Calculate distances between all pairs of sampled points
            for i in range(len(tracklog_subset)):
                for j in range(i + 1, len(tracklog_subset)):
                    p1 = tracklog_subset[i]
                    p2 = tracklog_subset[j]
                    distance = self.calculate_distance(
                        p1.lat, p1.lon, p2.lat, p2.lon)

                    if distance > max_distance:
                        max_distance = distance
                        point1_idx = sampled_indices[i]
                        point2_idx = sampled_indices[j]

            # Fine-tune around the identified maximum
            # Check points near the identified max distance points
            window = min(50, step)
            for i in range(max(0, point1_idx - window), min(len(self.tracklog), point1_idx + window + 1)):
                for j in range(max(0, point2_idx - window), min(len(self.tracklog), point2_idx + window + 1)):
                    if i != j:
                        p1 = self.tracklog[i]
                        p2 = self.tracklog[j]
                        distance = self.calculate_distance(
                            p1.lat, p1.lon, p2.lat, p2.lon)

                        if distance > max_distance:
                            max_distance = distance
                            point1_idx = i
                            point2_idx = j
        else:
            # For smaller tracklogs, check all pairs
            for i in range(len(self.tracklog)):
                for j in range(i + 1, len(self.tracklog)):
                    p1 = self.tracklog[i]
                    p2 = self.tracklog[j]
                    distance = self.calculate_distance(
                        p1.lat, p1.lon, p2.lat, p2.lon)

                    if distance > max_distance:
                        max_distance = distance
                        point1_idx = i
                        point2_idx = j

        point1 = self.tracklog[point1_idx]
        point2 = self.tracklog[point2_idx]

        return {
            'max_distance': max_distance,
            'point1_idx': point1_idx,
            'point2_idx': point2_idx,
            'point1': point1.to_dict(),
            'point2': point2.to_dict(),
            'time_difference': (point2.time - point1.time).total_seconds() if hasattr(point1.time, 'total_seconds') else None
        }

    def score_all_triangle_types(self) -> List[Dict]:
        """
        Score the flight for all triangle types in a single pass.

        Returns:
            List of scoring information dictionaries for each triangle type
        """
        if len(self.tracklog) < 5:
            return [self._empty_score(t) for t in ['closedFAI', 'FAI', 'closedFlat', 'flat']]

        # Check if the track is long enough for a valid triangle
        if len(self.tracklog) < 50:
            return [self._empty_score(t) for t in ['closedFAI', 'FAI', 'closedFlat', 'flat']]

        # Map hull indices to tracklog indices once
        hull_tracklog_indices = self.hull_tracklog_indices

        # Initialize best scores for each triangle type
        best_scores = {
            'closedFAI': {'score': 0, 'data': None},
            'FAI': {'score': 0, 'data': None},
            'closedFlat': {'score': 0, 'data': None},
            'flat': {'score': 0, 'data': None}
        }

        # Get multipliers for each triangle type
        multipliers = {
            triangle_type: self.scoring_rules.get(
                triangle_type, {}).get('multiplier', 1.0)
            for triangle_type in best_scores.keys()
        }

        """
        Should first calculate the longest triangle, then check if a longer FAI triangle exists. 
        After that, it should determine the XC start and end points.
        """
        max_perimeter = 0
        max_perimeter_triangle = None
        max_constrained_perimeter = 0
        max_constrained_perimeter_triangle = None
        max_constrained_perimeter_triangle_indices = None

        for tp_indices in itertools.combinations(hull_tracklog_indices, 3):

            tp1_idx, tp2_idx, tp3_idx = tp_indices
            # Ensure chronological order
            if not (tp1_idx < tp2_idx < tp3_idx):
                continue

            tp1, tp2, tp3 = (self.tracklog[i] for i in tp_indices)

            # Calculate the sides of the triangle once
            leg1 = self.calculate_distance(tp1.lat, tp1.lon, tp2.lat, tp2.lon)
            leg2 = self.calculate_distance(tp2.lat, tp2.lon, tp3.lat, tp3.lon)
            leg3 = self.calculate_distance(tp3.lat, tp3.lon, tp1.lat, tp1.lon)

            # Calculate perimeter
            perimeter = leg1 + leg2 + leg3

            # Find the longest and shortest sides
            shortest_leg = min(leg1, leg2, leg3)
            longest_leg = max(leg1, leg2, leg3)

            # Check for triangle with max perimeter
            if perimeter > max_perimeter:
                max_perimeter = perimeter
                max_perimeter_triangle_indices = tp_indices
                shortest_leg_max_perimeter = shortest_leg
                # print(max_perimeter)
                # print(tp_indices)

            # Check for triangle with max perimeter where shortest side is at least 8% of longest side
            if shortest_leg > 0.28 * perimeter and perimeter > max_constrained_perimeter:
                max_constrained_perimeter = perimeter
                max_constrained_perimeter_triangle_indices = tp_indices

        # checking if is possible to find a better FAI triangle inside the flat triangle *see documentation to understand how i got 0.24
        if (shortest_leg_max_perimeter/max_perimeter > 0.23):
            # print("checking for inside FAI")
            # Number of neighboring indices to consider (before and after the common vertex)
            num_neighbors = 7200  # Change this value as needed
            step = 1  # Change this value for a different step size

            # Extract the indices of the vertices of the maximum perimeter triangle
            tp1_idx, tp2_idx, tp3_idx = max_perimeter_triangle_indices

            # Compute the lengths of the triangle sides
            legs = {
                (tp1_idx, tp2_idx): self.calculate_distance(self.tracklog[tp1_idx].lat, self.tracklog[tp1_idx].lon,
                                                            self.tracklog[tp2_idx].lat, self.tracklog[tp2_idx].lon),
                (tp2_idx, tp3_idx): self.calculate_distance(self.tracklog[tp2_idx].lat, self.tracklog[tp2_idx].lon,
                                                            self.tracklog[tp3_idx].lat, self.tracklog[tp3_idx].lon),
                (tp3_idx, tp1_idx): self.calculate_distance(self.tracklog[tp3_idx].lat, self.tracklog[tp3_idx].lon,
                                                            self.tracklog[tp1_idx].lat, self.tracklog[tp1_idx].lon)
            }

            # Identify the two longest sides
            sorted_legs = sorted(
                legs.items(), key=lambda x: x[1], reverse=True)
            (longest_1, longest_2) = sorted_legs[:2]

            # The common vertex where the two longest sides meet
            common_vertex_idx = set(longest_1[0]) & set(longest_2[0])
            common_vertex_idx = list(common_vertex_idx)[
                0]  # Convert to integer

            # Identifica i due vertici che non sono il common vertex
            remaining_vertices = [
                idx for idx in max_perimeter_triangle_indices if idx != common_vertex_idx]

            # Find the neighboring indices around the common vertex with the given step size
            neighboring_indices_1 = [i for i in range(max(0, common_vertex_idx - num_neighbors * step),
                                                      min(len(
                                                          self.tracklog), common_vertex_idx + (num_neighbors + 1) * step),
                                                      step)]
            neighboring_indices_2 = [i for i in range(max(0, int(remaining_vertices[0] - num_neighbors / num_neighbors * step)),
                                     min(len(self.tracklog), int(
                                         remaining_vertices[0] + (num_neighbors / num_neighbors + 1) * step)),
                                     step)]
            neighboring_indices_3 = [i for i in range(max(0, int(remaining_vertices[1] - num_neighbors / num_neighbors * step)),
                                     min(len(self.tracklog), int(
                                         remaining_vertices[1] + (num_neighbors / num_neighbors + 1) * step)),
                                     step)]

            # Generate combinations with the neighboring indices
            for tp_indices_1 in neighboring_indices_1:
                for tp_indices_2 in neighboring_indices_2:
                    # Attempting to find a better point is useful, but in the end, optimization at the final stage is better
                    for tp_indices_3 in neighboring_indices_3:

                        tp1_idx, tp2_idx, tp3_idx = tp_indices_1, tp_indices_2, tp_indices_3

                        tp1_idx, tp2_idx, tp3_idx = sorted(
                            [tp1_idx, tp2_idx, tp3_idx])

                        tp1, tp2, tp3 = self.tracklog[tp1_idx], self.tracklog[tp2_idx], self.tracklog[tp3_idx]

                        # Calculate the sides of the triangle once
                        leg1 = self.calculate_distance(
                            tp1.lat, tp1.lon, tp2.lat, tp2.lon)
                        leg2 = self.calculate_distance(
                            tp2.lat, tp2.lon, tp3.lat, tp3.lon)
                        leg3 = self.calculate_distance(
                            tp3.lat, tp3.lon, tp1.lat, tp1.lon)

                        # Calculate perimeter
                        perimeter = leg1 + leg2 + leg3

                        # Find the longest and shortest sides
                        shortest_leg = min(leg1, leg2, leg3)
                        longest_leg = max(leg1, leg2, leg3)

                        # Check for triangle with max perimeter where shortest side is at least 8% of longest side
                        if shortest_leg > 0.28 * perimeter and perimeter > max_constrained_perimeter:
                            max_constrained_perimeter = perimeter
                            max_constrained_perimeter_triangle_indices = tp1_idx, tp2_idx, tp3_idx

        # Now, create best_triangle_indices dictionary with proper error checking
        best_triangle_indices = {}
        if max_perimeter_triangle_indices is not None:
            best_triangle_indices['max_perimeter'] = max_perimeter_triangle_indices
        if max_constrained_perimeter_triangle_indices is not None:
            best_triangle_indices['constrained_max_perimeter'] = max_constrained_perimeter_triangle_indices

        # Check all possible combinations of 3 turnpoints from the convex hull
        for triangle_type, tp_indices in best_triangle_indices.items():
            tp1_idx, tp2_idx, tp3_idx = tp_indices

            # Ensure chronological order
            if not (tp1_idx < tp2_idx < tp3_idx):
                continue

            tp1, tp2, tp3 = (self.tracklog[i] for i in tp_indices)

            # Calculate the sides of the triangle once
            leg1 = self.calculate_distance(tp1.lat, tp1.lon, tp2.lat, tp2.lon)
            leg2 = self.calculate_distance(tp2.lat, tp2.lon, tp3.lat, tp3.lon)
            leg3 = self.calculate_distance(tp3.lat, tp3.lon, tp1.lat, tp1.lon)

            start_idx, finish_idx = self._find_start_idx_finish_idx(
                tp1_idx, tp3_idx)
            start_point = self.tracklog[start_idx]
            finish_point = self.tracklog[finish_idx]

            # Calculate additional distances once
            start_to_tp1 = self.calculate_distance(
                start_point.lat, start_point.lon, tp1.lat, tp1.lon)
            tp3_to_finish = self.calculate_distance(
                tp3.lat, tp3.lon, finish_point.lat, finish_point.lon)

            # Calculate triangle perimeter
            perimeter = leg1 + leg2 + leg3

            # Skip invalid triangles
            if perimeter < 1:
                continue

            # Find shortest leg and ratio for FAI requirements
            shortest_leg = min(leg1, leg2, leg3)
            shortest_ratio = shortest_leg / perimeter

            # Calculate closing distance (from start to finish) once
            closing_distance = self.calculate_distance(
                start_point.lat, start_point.lon,
                finish_point.lat, finish_point.lon
            )

            # Calculate closing ratio once
            closing_ratio = closing_distance / perimeter

            # Common triangle data
            triangle_data = {
                'turnpoints': tp_indices,
                'turnpoints_data': [tp1.to_dict(), tp2.to_dict(), tp3.to_dict()],
                'take_off': self.take_off.to_dict(),
                'first_point': self.first_point.to_dict(),
                'start_point': start_point.to_dict(),
                'finish_point': finish_point.to_dict(),
                'last_point': self.last_point.to_dict(),
                'type': 'triangle',
                'properties': {
                    'leg1': leg1,
                    'leg2': leg2,
                    'leg3': leg3,
                    'start_to_tp1': start_to_tp1,
                    'tp3_to_finish': tp3_to_finish,
                    'tp1': tp1.to_dict(),
                    'tp2': tp2.to_dict(),
                    'tp3': tp3.to_dict(),
                    'shortest_leg': shortest_leg,
                    'shortest_leg_ratio': shortest_ratio,
                    'closing_distance': closing_distance,
                    'closing_ratio': closing_ratio
                }
            }

            # Check for each triangle type
            for triangle_type, best_score_data in best_scores.items():
                # Determine if this is a FAI triangle type
                is_fai = 'FAI' in triangle_type
                is_closed = 'closed' in triangle_type.lower()

                # Maximum allowed closing distance ratio
                max_closing_ratio = 0.05 if is_closed else 0.20

                # Skip if it doesn't meet FAI criteria
                if is_fai and shortest_ratio < 0.28:
                    continue

                # Skip if it doesn't meet closing criteria
                if closing_ratio > max_closing_ratio:
                    continue

                # Calculate score for this triangle type
                multiplier = multipliers[triangle_type]
                distance = perimeter - closing_distance
                score = distance * multiplier

                # Update best score if this one is better
                if score > best_score_data['score']:
                    # Create a copy of the common data
                    triangle_result = {
                        'score': score,
                        'triangle_type': triangle_type,
                        'turnpoints': tp_indices,
                        'turnpoints_index': [start_idx, tp1_idx, tp2_idx, tp3_idx, finish_idx],
                        'turnpoints_data': triangle_data['turnpoints_data'],
                        'take_off': self.take_off.to_dict(),
                        'first_point': triangle_data['first_point'],
                        'start_point': triangle_data['start_point'],
                        'finish_point': triangle_data['finish_point'],
                        'last_point': triangle_data['last_point'],
                        'type': 'triangle',
                        'properties': {
                            **triangle_data['properties'],
                            'total_distance': distance,
                            'multiplier': multiplier,
                            'closing_ratio': closing_ratio
                        }
                    }
                    best_scores[triangle_type] = {
                        'score': score, 'data': triangle_result}

        # Convert to list of results
        results = []
        for triangle_type, best_score_data in best_scores.items():
            if best_score_data['data']:
                results.append(best_score_data['data'])
            else:
                results.append(self._empty_score(triangle_type))

        return results

    def score_free_distance_flight(self, scoring_rules: Dict) -> Dict:
        """
        Score the flight as a free-distance flight (start->TP1->TP2->TP3->finish)
        using convex hull points as potential turnpoints.
        """
        if len(self.tracklog) < 5:
            return {
                'score': 0,
                'type': 'free_distance',
                'turnpoints': [],
                'route_points': [],
                'properties': {}
            }

        # Map hull indices to tracklog indices once
        hull_tracklog_indices = self.hull_tracklog_indices

        # Add a minimum distance from start/finish to ensure we don't select points too close
        min_end_dist = 5
        hull_tracklog_indices = [i for i in hull_tracklog_indices
                                 if (i >= min_end_dist and i <= len(self.tracklog) - min_end_dist)]

        best_score = 0
        best_indices = []
        best_distance = 0

        multiplier = scoring_rules.get('freeFlight', {}).get('multiplier', 1.0)

        for tp_indices in itertools.combinations(hull_tracklog_indices, 3):
            tp1_idx, tp2_idx, tp3_idx = tp_indices
            tp1, tp2, tp3 = self.tracklog[tp1_idx], self.tracklog[tp2_idx], self.tracklog[tp3_idx]

            # Make sure we're using the trackpoints in chronological order
            if not (tp1.time < tp2.time < tp3.time):
                continue

            start_to_tp1 = self.calculate_distance(
                self.first_point.lat, self.first_point.lon, tp1.lat, tp1.lon)
            leg1 = self.calculate_distance(tp1.lat, tp1.lon, tp2.lat, tp2.lon)
            leg2 = self.calculate_distance(tp2.lat, tp2.lon, tp3.lat, tp3.lon)
            tp3_to_finish = self.calculate_distance(
                tp3.lat, tp3.lon, self.last_point.lat, self.last_point.lon)

            total_distance = start_to_tp1 + leg1 + leg2 + tp3_to_finish
            score = total_distance * multiplier

            if score > best_score:
                best_score = score
                bestleg1 = leg1
                bestleg2 = leg2
                best_tp1_idx = tp1_idx
                best_tp3_idx = tp3_idx
                besttp1 = tp1
                besttp3 = tp3
                best_indices = tp_indices
                best_distance = total_distance

        # If no valid combination was found with the convex hull points,
        # we could fall back to the original grid-based method
        if not best_indices:
            # Build a small grid for free-distance checking as fallback
            grid_size = min(50, max(1, len(self.tracklog) // 20))
            step = max(1, len(self.tracklog) // grid_size)

            grid_points = [
                i for i in range(0, len(self.tracklog), step)
                if (i >= min_end_dist and i <= len(self.tracklog) - min_end_dist)
            ]

            for tp_indices in itertools.combinations(grid_points, 3):
                tp1_idx, tp2_idx, tp3_idx = tp_indices
                tp1, tp2, tp3 = self.tracklog[tp1_idx], self.tracklog[tp2_idx], self.tracklog[tp3_idx]

                # Make sure we're using the trackpoints in chronological order
                if not (tp1.time < tp2.time < tp3.time):
                    continue

                start_to_tp1 = self.calculate_distance(
                    self.first_point.lat, self.first_point.lon, tp1.lat, tp1.lon)
                leg1 = self.calculate_distance(
                    tp1.lat, tp1.lon, tp2.lat, tp2.lon)
                leg2 = self.calculate_distance(
                    tp2.lat, tp2.lon, tp3.lat, tp3.lon)
                tp3_to_finish = self.calculate_distance(
                    tp3.lat, tp3.lon, self.last_point.lat, self.last_point.lon)

                total_distance = start_to_tp1 + leg1 + leg2 + tp3_to_finish
                score = total_distance * multiplier

                if score > best_score:
                    best_score = score
                    bestleg1 = leg1
                    bestleg2 = leg2
                    best_tp1_idx = tp1_idx
                    best_tp3_idx = tp3_idx
                    besttp1 = tp1
                    besttp3 = tp3
                    best_indices = tp_indices
                    best_distance = total_distance

        # Optimization for stat and end point
        start_idx, finish_idx = self._find_best_end_for_free_distance_idx(
            best_tp1_idx, best_tp3_idx)
        finish_point = self.tracklog[finish_idx]
        start_point = self.tracklog[start_idx]

        # Recalcualate the score with the new start and end point
        start_to_tp1 = self.calculate_distance(
            start_point.lat, start_point.lon, besttp1.lat, besttp1.lon)
        tp3_to_finish = self.calculate_distance(
            besttp3.lat, besttp3.lon, finish_point.lat, finish_point.lon)

        total_distance = start_to_tp1 + bestleg1 + bestleg2 + tp3_to_finish
        score = total_distance * multiplier

        best_score = score
        best_distance = total_distance

        # If still no valid combination was found
        if not best_indices:
            return {
                'score': 0,
                'type': 'free_distance',
                'turnpoints': [],
                'route_points': [],
                'properties': {}
            }

        turnpoints_data = [self.tracklog[idx].to_dict()
                           for idx in best_indices]
        route_points = [self.first_point.to_dict()] + \
            turnpoints_data + [self.last_point.to_dict()]
        closing_distance = self.calculate_distance(
            self.first_point.lat, self.first_point.lon, self.last_point.lat, self.last_point.lon)
        turnpoints_index = [start_idx] + list(best_indices) + [finish_idx]

        return {
            'score': best_score,
            'turnpoints': list(best_indices),
            'turnpoints_index': turnpoints_index,
            'turnpoints_data': turnpoints_data,
            'take_off': self.take_off.to_dict(),
            'start_point': self.first_point.to_dict(),
            'finish_point': self.last_point.to_dict(),
            'closing_distance': closing_distance,
            'route_points': route_points,
            'type': 'free_distance',
            'properties': {
                'distance': best_distance,
                'multiplier': multiplier
            }
        }

    def score_out_return_flight(self, scoring_rules: Dict) -> Dict:
        """
        Score the flight as an out-and-return flight by finding the furthest point
        from the start point, using vectorized operations for efficiency.

        Returns:
            Dictionary with scoring information for out-and-return flight
        """
        if len(self.tracklog) < 5:
            return self._empty_score('out_return')

        # start_point = self.first_point

        # Get multiplier from scoring rules
        multiplier = scoring_rules.get('outReturn', {}).get('multiplier', 1.0)

        # Extract all lat/lon values into numpy arrays for vectorized calculation
        lats = np.array([p.lat for p in self.tracklog])
        lons = np.array([p.lon for p in self.tracklog])

        # Calculate all distances from start point to all points in one operation
        distances = self.haversine_distance_vectorized(
            self.first_point.lat,
            self.first_point.lon,
            lats,
            lons
        )

        # Find the furthest point
        furthest_idx = np.argmax(distances)
        max_distance = distances[furthest_idx]
        furthest_point = self.tracklog[furthest_idx]

        # Find the finish point (closest to start after reaching the furthest point)
        if furthest_idx >= len(self.tracklog) - 1:
            # If furthest point is the last point, use it as finish point
            finish_idx = furthest_idx
        else:
            # Calculate distances from start to all points after furthest_idx
            finish_distances = distances[furthest_idx:]
            # Find index of minimum distance relative to the slice
            min_idx_relative = np.argmin(finish_distances)
            # Convert back to original index
            finish_idx = furthest_idx + min_idx_relative

        finish_point = self.tracklog[finish_idx]

        # Calculate the missing distance
        missing_distance = self.calculate_distance(
            furthest_point.lat, furthest_point.lon,
            finish_point.lat, finish_point.lon
        )

        return {
            'out_distance': max_distance,
            'missing_distance': missing_distance,
            'furthest_point': furthest_point.to_dict()
        }

    def _find_best_end_for_free_distance_idx(self, tp1_idx: int, tp3_idx: int) -> tuple[int, int]:
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
        tp1_point = self.tracklog[tp1_idx]
        tp3_point = self.tracklog[tp3_idx]

        # Define the subsets
        subset_start = self.tracklog[:tp1_idx]
        subset_end = self.tracklog[tp3_idx:]

        # Initialize default values in case subsets are empty
        idx_farthest_from_tp1 = 0
        idx_farthest_from_tp3 = tp3_idx

        # Process the start subset if not empty
        if subset_start:
            # Convert geographic coordinates to radians
            points_rad = np.radians(
                np.array([(point.lat, point.lon) for point in subset_start]))
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
            points_rad = np.radians(
                np.array([(point.lat, point.lon) for point in subset_end]))
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

    def _find_start_idx_finish_idx(self, tp1_idx: int, tp3_idx: int) -> tuple:
        """
        Find the two closest points between the subset before tp1_idx and the subset after tp3_idx.

        Args:
            tp1_idx: Index defining the end of the first subset (exclusive)
            tp3_idx: Index defining the start of the second subset (inclusive)

        Returns:
            Tuple of indices (closest_start_idx, closest_end_idx) representing the closest points
            between the two subsets
        """
        # Extract subsets
        subset_start = self.tracklog[:tp1_idx]
        subset_end = self.tracklog[tp3_idx:]

        # Check if either subset is empty
        if not subset_start or not subset_end:
            return (0 if not subset_start else tp1_idx - 1,
                    tp3_idx if not subset_end else tp3_idx)

        # Convert geographic coordinates to radians for accuracy
        start_points_rad = np.radians(
            np.array([(point.lat, point.lon) for point in subset_start]))
        end_points_rad = np.radians(
            np.array([(point.lat, point.lon) for point in subset_end]))

        # Create KD-tree for the end subset for efficient nearest neighbor search
        tree = KDTree(end_points_rad)

        # Initialize variables to track minimum distance and corresponding indices
        min_distance = float('inf')
        closest_start_idx = None
        closest_end_idx = None

        # For each point in the start subset, find the closest point in the end subset
        for i, start_point_rad in enumerate(start_points_rad):
            distance, j = tree.query(start_point_rad, k=1)

            if distance < min_distance:
                min_distance = distance
                # Convert to original indices
                closest_start_idx = i
                closest_end_idx = j + tp3_idx

        # Ensure we have valid indices
        if closest_start_idx is None or closest_end_idx is None:
            # Fallback to first and last points if something went wrong
            return (0, tp3_idx)

        return (closest_start_idx, closest_end_idx)

    ########################################################################################

    def _find_closest_to_start_idx(self, tp3_idx: int) -> int:
        """
        Find the index of the point in the tracklog (starting from tp3_idx)
        that is closest to the start point using spatial indexing.
        """
        start_point = self.start_point
        subset = self.tracklog[tp3_idx:]

        if not subset:
            return tp3_idx

        # Convert geographic coordinates to radians for accuracy if using geographic coordinates
        # Skip this step if using Cartesian coordinates
        points_rad = np.radians(
            np.array([(point.lat, point.lon) for point in subset]))
        start_rad = np.radians([start_point.lat, start_point.lon])

        # Create KD-tree
        tree = KDTree(points_rad)

        # Query the tree for the nearest neighbor
        distance, idx = tree.query(start_rad, k=1)

        # Convert back to original index

        return idx + tp3_idx

    def _find_closest_to_end_idx(self, tp1_idx: int, finish_point) -> int:

        finish_point
        subset = self.tracklog[:tp1_idx]

        if not subset:
            return tp1_idx

        # Convert geographic coordinates to radians for accuracy if using geographic coordinates
        # Skip this step if using Cartesian coordinates
        points_rad = np.radians(
            np.array([(point.lat, point.lon) for point in subset]))
        end_rad = np.radians([finish_point.lat, finish_point.lon])

        # Create KD-tree
        tree = KDTree(points_rad)

        # Query the tree for the nearest neighbor
        distance, idx = tree.query(end_rad, k=1)

        # Convert back to original index
        return idx

    def _empty_score(self, flight_type: str) -> Dict:
        """Return an empty score object for the given flight type"""
        return {
            'score': 0,
            'type': flight_type,
            'turnpoints': [],
            'route_points': [],
            'properties': {}
        }

    def haversine_distance_vectorized(self, lat1, lon1, lat2, lon2):
        """
        Vectorized Haversine distance calculation.
        Can calculate distances from one point to many points efficiently.

        Args:
            lat1, lon1: Latitude and longitude of start point (scalar or array)
            lat2, lon2: Latitude and longitude of end points (arrays)

        Returns:
            Array of distances in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points in kilometers.
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r


class IGCParser:
    """Parser for IGC (International Gliding Commission) files"""

    def __init__(self, file_path: str):
        """Initialize with the path to an IGC file"""
        self.file_path = file_path
        self.points = []
        self.pilot_name = ""
        self.glider_type = ""
        self.date = None
        self.flight_metadata = {}

    def parse(self) -> List:
        """Parse the IGC file and return track points with necessary attributes"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
        except UnicodeDecodeError:
            # Try with latin-1 encoding if utf-8 fails
            with open(self.file_path, 'r', encoding='latin-1') as f:
                content = f.readlines()

        # Process each line
        for line in content:
            line = line.strip()
            if not line:
                continue

            record_type = line[0]

            # B records contain GPS points
            if record_type == 'B':
                point = self._parse_b_record(line)
                if point:
                    self.points.append(point)

            # H records contain header information
            elif record_type == 'H':
                self._parse_h_record(line)

        # Convert list of dict to list of objects with attributes
        tracklog = []
        for p in self.points:
            point = type('TrackPoint', (), {})()
            point.lat = p['lat']
            point.lon = p['lon']
            point.time = p['time']
            point.alt = p.get('gps_alt', p.get('pressure_alt', 0))

            # Add a to_dict method to the point object
            def to_dict_method(self):
                return {
                    'lat': self.lat,
                    'lon': self.lon,
                    'time': self.time.strftime('%H:%M:%S') if hasattr(self.time, 'strftime') else str(self.time),
                    'alt': self.alt
                }
            point.to_dict = to_dict_method.__get__(point)

            tracklog.append(point)

        return tracklog

    def _parse_b_record(self, line: str) -> Optional[Dict]:
        """Parse a B record (GPS fix)"""
        # Typical B record format:
        # B1111225310.9123N00108.8224WA00854F085

        if len(line) < 35:  # Minimum length for a valid B record
            return None

        try:
            # Parse time
            hours = int(line[1:3])
            minutes = int(line[3:5])
            seconds = int(line[5:7])
            time = datetime.time(hours, minutes, seconds)

            # Parse latitude
            lat_deg = int(line[7:9])
            lat_min = float(line[9:14]) / 1000.0
            latitude = lat_deg + lat_min / 60.0
            if line[14] == 'S':
                latitude = -latitude

            # Parse longitude
            lon_deg = int(line[15:18])
            lon_min = float(line[18:23]) / 1000.0
            longitude = lon_deg + lon_min / 60.0
            if line[23] == 'W':
                longitude = -longitude

            # Parse altitude (if available)
            pressure_alt = None
            gps_alt = None

            if len(line) >= 30:
                pressure_alt = int(line[25:30])

            if len(line) >= 35:
                gps_alt = int(line[30:35])

            return {
                'time': time,
                'lat': latitude,
                'lon': longitude,
                'pressure_alt': pressure_alt,
                'gps_alt': gps_alt
            }
        except (ValueError, IndexError):
            return None

    def _parse_h_record(self, line: str) -> None:
        """Parse a header record"""
        # Implementation similar to previous code


def process_igc_file(file_path: str, scoring_rules: Dict, optimization=False) -> Dict:
    """
    Process an IGC file and find the optimal scoring flight.

    Args:
        file_path: Path to the IGC file
        scoring_rules: Dictionary with multipliers for different flight types

    Returns:
        Dictionary with best scoring information
    """
    parser = IGCParser(file_path)
    tracklog = parser.parse()

    scorer = XCScorer(tracklog, scoring_rules)
    best_score = scorer.score_flight(optimization)

    return best_score


# Example usage:
if __name__ == "__main__":
    import sys
    import argparse

    config_xc = {
        # Scoring multipliers for different flight types
        "scoring_rules": {
            "flat": {
                "multiplier": 1.2
            },
            "FAI": {
                "multiplier": 1.4
            },
            "closedFAI": {
                "multiplier": 1.6
            },
            "closedFlat": {
                "multiplier": 1.4
            },
            "freeFlight": {
                "multiplier": 1.0
            }
        }
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Score paragliding flights using rules')
    parser.add_argument('igc_file', nargs='?',
                        help='Path to the IGC file to analyze')
    parser.add_argument('--optimization', action='store_true',
                        help='Enable track optimization to find the best possible score (increases processing time)')

    args = parser.parse_args()

    if args.igc_file:
        start_time = time.time()
        result = process_igc_file(
            args.igc_file, config_xc["scoring_rules"], args.optimization)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Best flight type: {result['type']}")
        if result['type'] == 'triangle':
            print(f"Triangle type: {result['triangle_type']}")
        print(f"Score: {result['score']:.2f} points")

        if 'total_distance' in result['properties']:
            print(f"Distance: {result['properties']['total_distance']:.2f} km")

        print(f"Multiplier: {result['properties']['multiplier']}")

        if result['type'] != 'free_distance' and 'closing_ratio' in result:
            print(f"Closing ratio: {result['closing_ratio'] * 100:.2f}%")

        # Print max distance information
        if 'max_distance_info' in result:
            max_info = result['max_distance_info']
            print("\nMaximum Distance Information:")
            print(f"Max distance: {max_info['max_distance']:.2f} km")
            p1 = max_info['point1']
            p2 = max_info['point2']
            print(
                f"Point 1: Lat {p1['lat']:.6f}, Lon {p1['lon']:.6f}, Time {p1['time']}")
            print(
                f"Point 2: Lat {p2['lat']:.6f}, Lon {p2['lon']:.6f}, Time {p2['time']}")
            if max_info['time_difference'] is not None:
                hours = max_info['time_difference'] // 3600
                minutes = (max_info['time_difference'] % 3600) // 60
                seconds = max_info['time_difference'] % 60
                print(
                    f"Time difference: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        if result.get('out_return'):
            print(
                f"\nOut distance: {result['out_return']['out_distance']:.2f} km")
            print(
                f"Missing distance: {result['out_return']['missing_distance']:.2f} km")
            furthest = result['out_return']['furthest_point']
            print(
                f"Furthest point: Lat {furthest['lat']:.6f}, Lon {furthest['lon']:.6f}, Alt {furthest.get('alt', 0)}")

        if result['turnpoints']:
            print("\nTurnpoints:")
            for i, tp in enumerate(result['turnpoints_data']):
                print(f"  TP{i+1}: Lat {tp['lat']:.6f}, Lon {tp['lon']:.6f}")

        # Start and finish points
        print(
            f"\nStart point: Lat {result['start_point']['lat']:.6f}, Lon {result['start_point']['lon']:.6f}")
        print(
            f"Finish point: Lat {result['finish_point']['lat']:.6f}, Lon {result['finish_point']['lon']:.6f}")
        print(
            f"Take-off point: Lat {result['take_off']['lat']:.6f}, Lon {result['take_off']['lon']:.6f}")

    else:
        parser.print_help()
