# XC Score: Paragliding/Hang-gliding Cross-Country Flight Scoring System

## Overview

XC Score is a sophisticated system for analyzing and scoring paragliding and hang-gliding cross-country (XC) flights according to standard competition rules. It processes IGC flight logs to identify the best possible flight path and calculate scores based on different flight types.

## Table of Contents

- [Flight Types and Scoring Rules](#flight-types-and-scoring-rules)
- [Technical Implementation](#technical-implementation)
  - [Distance Calculation](#distance-calculation)
  - [Takeoff Detection](#takeoff-detection)
  - [Flight Analysis Pipeline](#flight-analysis-pipeline)
  - [Triangle Detection](#triangle-detection)
  - [Free Distance Flight Scoring](#free-distance-flight-scoring)
  - [Out-and-Return Flight Scoring](#out-and-return-flight-scoring)
- [Optimization Methodology](#optimization-methodology)
  - [Triangle Optimization](#triangle-optimization)
  - [Free Distance Optimization](#free-distance-optimization)
  - [Spatial Indexing](#spatial-indexing)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Programmatic Usage](#programmatic-usage)
- [Configuration](#configuration)

## Flight Types and Scoring Rules

The system scores flights according to the following recognized flight types, each with its own rules and multipliers:

### 1. Triangle Flights

Triangular routes are scored based on perimeter distance minus the closing distance (distance between start and finish points). The following triangle types are recognized:

- **FAI Triangle**: A triangle where the shortest side is at least 28% of the total perimeter. Multiplier: 1.4
- **Flat Triangle**: A triangle with no constraints on side lengths. Multiplier: 1.2
- **Closed FAI Triangle**: An FAI triangle where the starting and finishing points are close (within 5% of triangle perimeter). Multiplier: 1.6
- **Closed Flat Triangle**: A flat triangle where the starting and finishing points are close (within 5% of triangle perimeter). Multiplier: 1.4

### 2. Free Distance Flight

A route with up to 3 turnpoints designed to maximize total distance. The score is calculated as the sum of the legs between start → TP1 → TP2 → TP3 → finish. Multiplier: 1.0

### 3. Out-and-Return Flight

A flight from the start point to a furthest point and back. The score is based on the distance to the furthest point reached. Multiplier: 1.2

## Technical Implementation

### Distance Calculation

All distances are calculated using the Haversine formula, which determines the great-circle distance between two points on a sphere (Earth) given their longitudes and latitudes:

```python
def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r
```

For efficiency when calculating distances to multiple points, a vectorized version using NumPy is implemented:

```python
def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r
```

### Takeoff Detection

The system automatically detects the actual takeoff point by analyzing speed changes and altitude patterns in the tracklog. This is crucial for accurate flight scoring, as the actual flight begins at takeoff, not where the recording started. The algorithm:

1. Calculates speeds between consecutive points
2. Applies exponential moving average smoothing to reduce noise
3. Detects takeoff based on several criteria:
   - Speed increase of at least 7 km/h
   - Sustained speed above 10 km/h
   - Speed ratio increase of at least 2x
   - Height-speed product (indicating climbing) above threshold
   - Consistent movement for 5+ consecutive points

```python
def _detect_takeoff_point(self):
    # Algorithm parameters
    MIN_TAKEOFF_SPEED = 10.0       # km/h
    MIN_SPEED_INCREASE = 7.0       # km/h increase
    TAKEOFF_SPEED_RATIO = 2.0      # Speed multiplier during takeoff
    MIN_HEIGHT_SPEED_PRODUCT = 20.0
    CONSECUTIVE_POINTS = 5         # Number of points to confirm consistent flight

    # Implementation checks for a clear pattern of increased speed and altitude change
    # that marks the transition from ground handling to flight
    # ...
```

### Flight Analysis Pipeline

The scoring process follows these steps:

1. Parse the IGC file to extract tracklog data (timestamps, coordinates, altitude)
2. Detect the actual takeoff point
3. Compute track curvature to determine how many segments to use in convex hull calculation
4. Calculate convex hull points to reduce computational complexity
5. Score the flight for all possible flight types:
   - Multiple triangle configurations
   - Free distance with 3 turnpoints
   - Out-and-return
6. Calculate maximum distance between any two points in the track
7. Select the highest scoring flight type
8. Optionally apply optimization algorithms to refine the score

### Triangle Detection

Triangle detection uses a sophisticated approach:

1. **Convex Hull Filtering**: The tracklog is divided into subsets, and a convex hull algorithm identifies potential turnpoint candidates, significantly reducing the search space.

2. **Efficient Triangle Selection**: Two strategies are used to find the best triangles:

   - Max perimeter triangle from convex hull points
   - Max perimeter triangle where the shortest side is at least 28% of perimeter (FAI triangle)

3. **Start/Finish Point Optimization**: For each triangle, optimal start and finish points are determined to maximize the scored distance while respecting closing distance requirements.

4. **Triangle Classification**: Each triangle is classified as FAI or flat, and closed or open:

   ```python
   # FAI criteria: shortest leg at least 28% of perimeter
   is_fai = shortest_ratio >= 0.28

   # Closed triangle criteria: closing distance less than 5% of perimeter
   is_closed = closing_ratio <= 0.05
   ```

5. **Special FAI Detection Algorithm**: For triangles close to FAI criteria, an additional intensive search is performed:
   ```python
   # If the shortest leg is close to FAI criteria (>23% of perimeter)
   if (shortest_leg_max_perimeter/max_perimeter > 0.23):
       # Additional search around triangle vertices to find a better FAI triangle
       # ...
   ```

### Free Distance Flight Scoring

The free distance scoring algorithm:

1. Uses convex hull points as potential turnpoints to reduce computational load
2. Evaluates all chronologically valid combinations of 3 turnpoints
3. Calculates total distance as: start → TP1 → TP2 → TP3 → finish
4. Optimizes start and finish points by finding the farthest points from TP1 and TP3
5. Falls back to a grid-based search if no valid combination is found from convex hull points

```python
# Key distance calculation
total_distance = start_to_tp1 + leg1 + leg2 + tp3_to_finish
score = total_distance * multiplier
```

### Out-and-Return Flight Scoring

The out-and-return algorithm:

1. Calculates distances from the starting point to all points in the tracklog using vectorized operations
2. Finds the furthest point from the start
3. Identifies the best finish point (closest to start after reaching the furthest point)
4. Calculates the missing distance (how far the pilot is from completing a perfect out-and-return)

## Optimization Methodology

When the `--optimization` flag is used, additional algorithms are applied to refine the initial flight score.

### Triangle Optimization

The triangle optimization performs a local search around each turnpoint to find a better triangle:

1. Creates a search radius around each original turnpoint
2. Tests all valid combinations of points within these radii
3. For FAI triangles, maintains the 28% shortest leg requirement
4. Recalculates score with the optimized triangle

```python
def optimize_track_triangle(tracklog, best_score_info, search_radius=1):
    # If FAI triangle, maintain the 28% shortest leg requirement
    if 'FAI' in best_score_info.get('triangle_type', ''):
        ratio = 0.28
    else:
        ratio = 0

    # Search neighborhood around each turnpoint within search_radius
    range_tp1 = range(max(0, tp1_idx - search_radius), min(len(tracklog), tp1_idx + search_radius + 1))
    range_tp2 = range(max(0, tp2_idx - search_radius), min(len(tracklog), tp2_idx + search_radius + 1))
    range_tp3 = range(max(0, tp3_idx - search_radius), min(len(tracklog), tp3_idx + search_radius + 1))

    # Test all combinations and find the best one
    for i1, i2, i3 in itertools.product(range_tp1, range_tp2, range_tp3):
        # Check if the new triangle is better and meets requirements
        # ...
```

### Free Distance Optimization

For free distance flights, the optimization:

1. Searches for better turnpoint combinations within a radius of each original turnpoint
2. Recalculates start and finish points for maximum distance
3. Updates the final score based on the optimized route

### Spatial Indexing

The system uses KD-Trees for efficient spatial queries:

1. **Nearest Neighbor Searches**: For finding closest points between start and finish areas
2. **Farthest Point Detection**: For finding optimal start/finish points in free distance flights
3. **Out-and-Return Calculation**: For efficiently finding the farthest point from start

```python
# Example of KD-Tree usage for spatial indexing
tree = KDTree(points_rad)
distances, indices = tree.query(reference_point_rad, k=len(points))
```

## Usage

### Command Line Interface

```bash
# Basic usage
python xc_scorer.py path/to/flight.igc

# With optimization (better results but slower processing)
python xc_scorer.py --optimization path/to/flight.igc

# Show help
python xc_scorer.py -h
```

### Programmatic Usage

```python
from xc_scorer import XCScorer, IGCParser

# Define scoring rules
scoring_rules = {
    "flat": {"multiplier": 1.2},
    "FAI": {"multiplier": 1.4},
    "closedFAI": {"multiplier": 1.6},
    "closedFlat": {"multiplier": 1.4},
    "freeFlight": {"multiplier": 1.0},
    "outReturn": {"multiplier": 1.2}
}

# Parse IGC file
parser = IGCParser('flight.igc')
tracklog = parser.parse()

# Score the flight
scorer = XCScorer(tracklog, scoring_rules)
result = scorer.score_flight(optimization=True)  # Set to True for optimization

# Access results
print(f"Score: {result['score']:.2f} points")
print(f"Flight type: {result['type']}")
```

## Configuration

Scoring rules can be customized by adjusting the multipliers:

```python
config_xc = {
    "scoring_rules": {
        "flat": {"multiplier": 1.2},
        "FAI": {"multiplier": 1.4},
        "closedFAI": {"multiplier": 1.6},
        "closedFlat": {"multiplier": 1.4},
        "freeFlight": {"multiplier": 1.0},
        "outReturn": {"multiplier": 1.2}
    }
}
```

Additional algorithm parameters that can be adjusted include:

- Convex hull calculation parameters
- FAI triangle criteria (currently 28%)
- Closed triangle criteria (currently 5%)
- Takeoff detection parameters
- Optimization search radius

## Contributors
- Simone Severini - [@simseve](https://github.com/simseve)

Special thanks for outstanding contribution:
- Vittorio Rena - [@virking72](https://github.com/virking72)
- Max Ghelfi - [@MaxNero](https://github.com/MaxNero)

## Hike and Fly Scoring System

See implementation example: [https://www.hikeandfly.app/tools/xcscore](https://www.hikeandfly.app/tools/xcscore)