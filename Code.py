# Importing required libraries
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import trange

# Constants
G = 6.673e-11                 # Gravitational Constant
AU = 1.496e11                 # Astronomical Unit in m
YEAR = 365*24*60*60.0         # Seconds in one year
MM = 6e24                     # Normalizing mass
ME = 6e24/MM                  # Normalized mass of Earth
MS = 2e30/MM                  # Normalized mass of Sun
MJ = 500*1.9e27/MM            # Normalized mass of Jupiter
GG = (MM*G*YEAR**2)/(AU**3)   # Gravitational constant for simulation

# Function to calculate gravitational force
def gravitational_force(m1: float, m2: float, r: np.ndarray) -> np.ndarray:
    F_mag = GG * m1 * m2 / (np.linalg.norm(r) + 1e-20)**2
    theta = np.arctan2(np.abs(r[1]), np.abs(r[0]) + 1e-20)
    F = F_mag * np.array([np.cos(theta), np.sin(theta)])
    F *= -np.sign(r)
    return F

# RK4 Solver
def RK4Solver(t: float, r: np.ndarray, v: np.ndarray, h: float, planet: str, r_other: np.ndarray, v_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def dr_dt(v: np.ndarray) -> np.ndarray:
        return v

    def dv_dt(r: np.ndarray, planet: str) -> np.ndarray:
        if planet == 'earth':
            return (gravitational_force(ME, MS, r) + gravitational_force(ME, MJ, r - r_other)) / ME
        elif planet == 'jupiter':
            return (gravitational_force(MJ, MS, r) - gravitational_force(MJ, ME, r - r_other)) / MJ

    k11 = dr_dt(v)
    k21 = dv_dt(r, planet)

    k12 = dr_dt(v + 0.5 * h * k21)
    k22 = dv_dt(r + 0.5 * h * k11, planet)

    k13 = dr_dt(v + 0.5 * h * k22)
    k23 = dv_dt(r + 0.5 * h * k12, planet)

    k14 = dr_dt(v + h * k23)
    k24 = dv_dt(r + h * k13, planet)

    y0 = r + h * (k11 + 2 * k12 + 2 * k13 + k14) / 6
    y1 = v + h * (k21 + 2 * k22 + 2 * k23 + k24) / 6

    return y0, y1

# Setup animation
def setup_animation() -> Tuple[plt.Figure, plt.Axes, plt.Line2D, plt.Line2D, plt.Text]:
    fig, ax = plt.subplots()
    ax.axis('square')
    ax.set_xlim((-7.2, 7.2))
    ax.set_ylim((-7.2, 7.2))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.plot(0, 0, 'o', markersize=9, markerfacecolor="#FDB813", markeredgecolor="#FD7813")

    line_earth, = ax.plot([], [], 'o-', color='#d2eeff', markevery=10000, markerfacecolor='#0077BE', lw=2)
    line_jupiter, = ax.plot([], [], 'o-', color='#e3dccb', markersize=8, markerfacecolor='#f66338', lw=2, markevery=10000)
    ttl = ax.text(0.24, 1.05, '', transform=ax.transAxes, va='center')

    return fig, ax, line_earth, line_jupiter, ttl

# Animation function
def animate(i: int) -> Tuple[plt.Line2D, plt.Line2D, plt.Text]:
    earth_trail, jupiter_trail = 40, 200
    tm_yr = 'Elapsed time = ' + str(round(t[i], 1)) + ' years'
    ttl.set_text(tm_yr)
    line_earth.set_data(r[i:max(1, i - earth_trail):-1, 0], r[i:max(1, i - earth_trail):-1, 1])
    line_jupiter.set_data(r_jupiter[i:max(1, i - jupiter_trail):-1, 0], r_jupiter[i:max(1, i - jupiter_trail):-1, 1])
    return line_earth, line_jupiter, ttl

# Initialization
ti, tf = 0, 120  # Initial and final time in years
N = 100 * tf     # 100 points per year
t = np.linspace(ti, tf, N)  # Time array
h = t[1] - t[0]  # Time step

# Position and Velocity Initialization
r = np.zeros([N, 2])         # Position of Earth
v = np.zeros([N, 2])         # Velocity of Earth
r_jupiter = np.zeros([N, 2])  # Position of Jupiter
v_jupiter = np.zeros([N, 2])  # Velocity of Jupiter

# User input for custom planet
use_default = input("Use default values? (yes/no): ").strip().lower()

if use_default == 'yes':
    # Initial Conditions for default simulation
    r[0] = [1496e8 / AU, 0]
    r_jupiter[0] = [5.2, 0]
    v[0] = [0, np.sqrt(MS * GG / r[0, 0])]
    v_jupiter[0] = [0, 13.06e3 * YEAR / AU]
else:
    # User input for custom planet
    custom_mass = float(input("Enter the mass of the custom planet (in kg): "))
    custom_distance = float(input("Enter the initial distance from the Sun (in meters): "))
    custom_velocity = float(input("Enter the initial velocity of the custom planet (in meters/second): "))

    # Normalized mass of the custom planet
    MC = custom_mass / MM

    # Initial Conditions for custom simulation
    r[0] = [custom_distance / AU, 0]
    v[0] = [0, custom_velocity * YEAR / AU]

    # Jupiter's initial conditions remain the same
    r_jupiter[0] = [5.2, 0]
    v_jupiter[0] = [0, 13.06e3 * YEAR / AU]

# Running the simulation
for i in trange(N - 1, desc="Generating Animation"):
    r[i + 1], v[i + 1] = RK4Solver(t[i], r[i], v[i], h, 'earth', r_jupiter[i], v_jupiter[i])
    r_jupiter[i + 1], v_jupiter[i + 1] = RK4Solver(t[i], r_jupiter[i], v_jupiter[i], h, 'jupiter', r[i], v[i])

# Setting Up the Animation
fig, ax, line_earth, line_jupiter, ttl = setup_animation()
ax.plot([-6,-5],[6.5,6.5],'r-')
ax.text(-4.5,6.3,r'1 AU = $1.496 \times 10^8$ km')

ax.plot(-6,-6.2,'o', color = '#d2eeff', markerfacecolor = '#0077BE')
ax.text(-5.5,-6.4,'Earth')

ax.plot(-3.3,-6.2,'o', color = '#e3dccb',markersize = 8, markerfacecolor = '#f66338')
ax.text(-2.9,-6.4,'Super Jupiter (500x mass)')

ax.plot(5,-6.2,'o', markersize = 9, markerfacecolor = "#FDB813",markeredgecolor ="#FD7813")
ax.text(5.5,-6.4,'Sun')

# Creating the Animation
anim = animation.FuncAnimation(fig, animate, frames=4000, interval=1, blit=False)

# Displaying the Animation
plt.show()

# Moon Simulation
# Initialize masses
m1 = 5.972e24  # Mass of the Earth in kg
m2 = 7.348e22  # Mass of the Moon in kg

# Initial positions and velocities
r1 = np.array([0, 0])  # Position of Earth
r2 = np.array([3.844e8, 0])  # Position of Moon (3.844e8 m = distance from Earth to Moon)
v1 = np.array([0, 0])  # Velocity of Earth
v2 = np.array([0, 1022])  # Velocity of Moon (approx speed of Moon)

# Time parameters
h = 60  # Time step in seconds
