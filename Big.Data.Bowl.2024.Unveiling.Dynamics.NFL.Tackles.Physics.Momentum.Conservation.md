# Unveiling the Dynamics of NFL Tackles and the Physics of Momentum Conservation

# Code

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_analyze_data():
    """Load and analyze tackling data from a CSV file."""
    df = pd.read_csv('tackles.csv')
    print("First few rows of the DataFrame:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    player_stats = df.groupby('nflId').agg({
        'tackle': 'sum',
        'assist': 'sum',
        'forcedFumble': 'sum',
        'pff_missedTackle': 'sum'
    }).reset_index()
    print("\nPlayer statistics:")
    print(player_stats)
    visualize_data(player_stats)

def visualize_data(player_stats):
    """Visualize the tackling data."""
    plt.figure(figsize=(10, 6))
    plt.bar(player_stats['nflId'].astype(str), player_stats['tackle'])
    plt.xlabel('NFL ID')
    plt.ylabel('Number of Tackles')
    plt.title('Number of Tackles by Player')
    plt.xticks(rotation=45)
    plt.show()

def validate_inputs(m1, v1, m2, v2):
    """Validates the input parameters for momentum calculation."""
    if m1 < 0 or m2 < 0 or v1 < 0 or v2 < 0:
        raise ValueError("Mass and velocity must be non-negative.")

def calculate_momentum_conservation(m1, v1, theta1, m2, v2, theta2, elastic=True):
    """Calculates the final velocities and angles of two colliding objects."""
    validate_inputs(m1, v1, m2, v2)
    theta1, theta2 = np.radians(theta1), np.radians(theta2)
    p1_before = m1 * v1 * np.array([np.cos(theta1), np.sin(theta1)])
    p2_before = m2 * v2 * np.array([np.cos(theta2), np.sin(theta2)])
    p_tot_before = p1_before + p2_before
    if not elastic:
        p_tot_before *= (1 - 0.5)  # Coefficient of restitution for inelastic collisions
    p1_after = p_tot_before * (m1 / (m1 + m2))
    p2_after = p_tot_before * (m2 / (m1 + m2))
    v1f = np.linalg.norm(p1_after) / m1
    v2f = np.linalg.norm(p2_after) / m2
    theta1f = np.degrees(np.arctan2(p1_after[1], p1_after[0]))
    theta2f = np.degrees(np.arctan2(p2_after[1], p2_after[0]))
    return ((v1f, theta1f), (v2f, theta2f))

if __name__ == "__main__":
    # Data Analysis
    load_and_analyze_data()

    # Physics Simulation
    m1, v1, theta1 = 10, 10, 30  # First object
    m2, v2, theta2 = 5, 5, 60    # Second object
    (v1f, theta1f), (v2f, theta2f) = calculate_momentum_conservation(m1, v1, theta1, m2, v2, theta2)
    print(f"\nFinal velocities and angles:")
    print(f"Object 1: v1f = {v1f:.2f} m/s, theta1f = {theta1f:.2f} degrees")
    print(f"Object 2: v2f = {v2f:.2f} m/s, theta2f = {theta2f:.2f} degrees")

```


## The Data: NFL Tackles

We start by examining a dataset that contains information about tackles made during NFL games. The dataset includes various metrics such as the game ID, play ID, NFL ID of the player, number of tackles, assists, forced fumbles, and missed tackles. Here's a glimpse of the data:

```plaintext
First few rows of the DataFrame:
       gameId  playId  nflId  tackle  assist  forcedFumble  pff_missedTackle
0  2022090800     101  42816       1       0             0                 0
1  2022090800     393  46232       1       0             0                 0
2  2022090800     486  40166       1       0             0                 0
3  2022090800     646  47939       1       0             0                 0
4  2022090800     818  40107       1       0             0                 0
```

### Basic Statistics

The dataset contains 17,426 records with the following basic statistics:

- Mean number of tackles: 0.569
- Mean number of assists: 0.315
- Mean number of forced fumbles: 0.006
- Mean number of missed tackles: 0.120

### Player Statistics

We also aggregated the data by player (NFL ID) to get the total number of tackles, assists, forced fumbles, and missed tackles for each player. The data reveals a wide range of performance metrics among the players.

## The Physics: Conservation of Momentum

Switching gears, let's talk about the physics of collisions, specifically the conservation of momentum. When two objects collide, their total momentum before and after the collision remains constant, provided no external forces are acting on them.

### Simulation Results

We simulated a collision between two objects with the following initial conditions:

- Object 1: 10 kg, 10 m/s, 30 degrees
- Object 2: 5 kg, 5 m/s, 60 degrees

The final velocities and angles were:

- Object 1: v1f = 8.15 m/s, theta1f = 35.87 degrees
- Object 2: v2f = 8.15 m/s, theta2f = 35.87 degrees
