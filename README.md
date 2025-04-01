# XContestScorer

## Overview
XContestScorer is a Python library for scoring paragliding flights according to XContest rules. It analyzes GPS tracklogs to identify and score different flight types, including triangles (FAI and flat), free distance flights, and out-and-return routes.

## Features
- **Multiple Flight Type Scoring**: Automatically calculates scores for different flight types:
  - FAI triangles
  - Flat triangles
  - Closed FAI triangles
  - Closed flat triangles
  - Free distance flights (with up to 3 turnpoints)
  - Out-and-return flights
- **Takeoff Detection**: Automatically identifies the actual takeoff point by analyzing speed and altitude changes
- **Optimization Algorithms**: Uses computational geometry (convex hull) and spatial indexing for efficient flight path analysis
- **Maximum Distance Calculation**: Identifies the maximum distance between any two points in the tracklog
- **IGC File Support**: Built-in parser for standard IGC (International Gliding Commission) flight log files
- **DataFrame Support**: Can work with pandas DataFrames containing tracklog data

## Installation
```bash
pip install xcontestscorer
```

## Usage

### Basic Example
```python
from xcontestscorer import XContestScorer, IGCParser

# Define scoring rules
scoring_rules = {
    "flat": {"multiplier": 1.2},
    "FAI": {"multiplier": 1.4},
    "closedFAI": {"multiplier": 1.6},
    "closedFlat": {"multiplier": 1.4},
    "freeFlight": {"multiplier": 1.0}
}

# Process an IGC file
def process_flight(file_path):
    # Parse the IGC file
    parser = IGCParser(file_path)
    tracklog = parser.parse()
    
    # Score the flight
    scorer = XContestScorer(tracklog, scoring_rules)
    result = scorer.score_flight()
    
    return result

# Example
result = process_flight("path/to/flight.igc")
print(f"Score: {result['score']:.2f} points")
print(f"Flight type: {result['type']}")
```

### Working with pandas DataFrames
```python
import pandas as pd
from xcontestscorer import XContestScorer

# Load tracklog data into a DataFrame
df = pd.read_csv("flight_data.csv")

# Ensure DataFrame has 'lat', 'lon', 'datetime', and altitude columns
# Initialize scorer
scorer = XContestScorer(df, scoring_rules)
result = scorer.score_flight()
```

## Command Line Interface
The script can be run directly from the command line:

```bash
python xcontest_scorer.py path/to/flight.igc
```

Output will include:
- Best flight type
- Score in points
- Distance in kilometers
- Multiplier used
- Closing ratio (for triangle flights)
- Maximum distance information
- Turnpoint details

## Configuration
You can customize the scoring rules by modifying the multipliers:

```python
config_xc = {
    "scoring_rules": {
        "flat": {"multiplier": 1.2},
        "FAI": {"multiplier": 1.4},
        "closedFAI": {"multiplier": 1.6},
        "closedFlat": {"multiplier": 1.4},
        "freeFlight": {"multiplier": 1.0},
        "outReturn": {"multiplier": 1.2}  # Optional
    }
}
```

## Technical Details

### Flight Types
- **FAI Triangle**: Triangle where the shortest leg is at least 28% of the total perimeter
- **Flat Triangle**: Triangle with no constraints on leg lengths
- **Closed Triangle**: Triangle where the start and finish points are close (within 5% of perimeter)
- **Free Distance**: Route with up to 3 turnpoints chosen to maximize distance
- **Out and Return**: Flight from start to the furthest point and back

### Algorithms
- Convex hull is used to identify potential turnpoints efficiently
- KD-Tree spatial indexing accelerates nearest-neighbor searches
- Vectorized operations with NumPy for fast distance calculations
- Haversine formula for accurate distance calculation on the Earth's surface

## Requirements
- Python 3.6+
- NumPy
- SciPy
- Pandas (optional, for DataFrame support)
- Logging

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.