import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpmath import jtheta

# --- 1. Isolated Vectorization of Jacobi Theta ---

def _jtheta_3_complex(z, q):
    """Evaluates the 3rd Jacobi Theta function and casts to standard complex."""
    return complex(jtheta(3, z, q))

# Create a ufunc that takes 2 inputs (z, q) and returns 1 output
_vectorized_jtheta3 = np.frompyfunc(_jtheta_3_complex, 2, 1)


# --- 2. 1D Wave Function Evaluation (NumPy Native) ---

def psi_1D_infinite_well(x, t, x0, p0, sigma0, L, m=1.0, hbar=1.0):
    """
    Evaluates the 1D wave packet using fast NumPy array math, 
    calling out to mpmath only for the theta function evaluation.
    """
    # Complex spreading parameter
    gamma = 1.0 / (4 * sigma0**2 * (1 + 1j * (hbar * t) / (2 * m * sigma0**2)))
    
    # Classical trajectory and wave number
    xc = x0 + (p0 / m) * t
    k0 = p0 / hbar
    
    # Theta function parameters using np.exp
    q = np.exp(-4 * gamma * L**2)
    z_plus  = -2j * gamma * L * (x - xc) - k0 * L
    z_minus = -2j * gamma * L * (x + xc) + k0 * L
    
    # Free-space envelopes using native np.exp
    env_plus  = np.exp(-gamma * (x - xc)**2 + 1j * k0 * (x - (p0 * t)/(2*m)))
    env_minus = np.exp(-gamma * (x + xc)**2 - 1j * k0 * (x + (p0 * t)/(2*m)))
    
    # Evaluate Theta functions and immediately cast to native complex arrays
    theta_plus  = np.asarray(_vectorized_jtheta3(z_plus, q), dtype=np.complex128)
    theta_minus = np.asarray(_vectorized_jtheta3(z_minus, q), dtype=np.complex128)
    
    # Combine using fast NumPy arithmetic
    psi = env_plus * theta_plus - env_minus * theta_minus
    
    return psi


# --- 3. 2D Setup & Physical Parameters ---

# Physics parameters
L_x, L_y = 10.0, 10.0
x0, y0 = 3.0, 3.0           # Start in the lower left quadrant
p0x, p0y = 4.0, 3.0         # Angled trajectory
sigma0 = 1
m = 1.0
hbar = 1.0

# Resolution for the 2D grid
N_x, N_y = 200, 200
x_grid = np.linspace(0, L_x, N_x)
y_grid = np.linspace(0, L_y, N_y)

# Figure setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("2D Wave Packet in an Infinite Well\n(Domain Coloring: Phase & Amplitude)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Blank RGB array for initialization
blank_rgb_frame = np.zeros((N_y, N_x, 3))

im = ax.imshow(
    blank_rgb_frame, 
    extent=[0, L_x, 0, L_y], 
    origin='lower', 
    animated=True
)

# --- 4. Animation Logic ---

vmax_brightness = 1.2 # Adjust to change how bright the peaks appear

def init():
    """Initializes the empty frame."""
    im.set_array(blank_rgb_frame)
    return [im]

def update(frame):
    """Updates the frame data."""
    t = frame
    
    # Evaluate 1D arrays
    psi_x = psi_1D_infinite_well(x_grid, t, x0, p0x, sigma0, L_x, m, hbar)
    psi_y = psi_1D_infinite_well(y_grid, t, y0, p0y, sigma0, L_y, m, hbar)
    
    # Construct 2D complex wave function via broadcasting
    psi_2d = psi_y[:, np.newaxis] * psi_x[np.newaxis, :]
    
    # Phase mapped to Hue [0, 1]
    phase = np.angle(psi_2d)
    h = (phase + np.pi) / (2 * np.pi)
    
    # Modulus mapped to Value/Brightness [0, 1]
    mag = np.abs(psi_2d)
    v = np.clip(mag / vmax_brightness, 0, 1)
    
    # Full Saturation
    s = np.ones_like(h)
    
    # Stack into HSV and convert to RGB
    hsv_image = np.dstack((h, s, v))
    rgb_image = mcolors.hsv_to_rgb(hsv_image)
    
    # Update image array
    im.set_array(rgb_image)
    return [im]


# --- 5. Execution & Saving ---

# Time array: Run from t=0 to t=10 with 300 frames
time_frames = np.linspace(0, 10, 300)
fps = 30

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=time_frames, 
    init_func=init, 
    blit=True, 
    interval=1000/fps
)

# Set to True to save the animation, False to view interactively
SAVE_ANIMATION = True
SAVE_FORMAT = 'mp4' # 'mp4' or 'gif'

if SAVE_ANIMATION:
    print(f"Saving animation as {SAVE_FORMAT}...")
    if SAVE_FORMAT == 'mp4':
        writer = animation.FFMpegWriter(fps=fps, bitrate=1000)
        ani.save("wave_packet_2D.mp4", writer=writer)
        print("Saved to wave_packet_2D.mp4")
    elif SAVE_FORMAT == 'gif':
        writer = animation.PillowWriter(fps=fps)
        ani.save("wave_packet_2D.gif", writer=writer)
        print("Saved to wave_packet_2D.gif")
else:
    plt.tight_layout()
    plt.show()