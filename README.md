# Quaint

## What is this program?
This is a Python-based desktop application that visualizes the time-dependent 2D Schrödinger equation in 3D. Built with **PyQt6**, **PyQtGraph**, and **SciPy**, it provides a real-time, interactive environment to explore quantum mechanics, specifically focusing on how wave packets interact with various potential energy landscapes.

## What does it do?
The application simulates and animates the evolution of a 2D Gaussian wave packet over time. Features include:
*   **Custom Potentials:** Use a built-in canvas to brush or erase custom potential energy barriers, or load mathematical presets (e.g., Harmonic Oscillator, Gaussian Bump).
*   **Initial Conditions:** Visually set the wave packet's starting position ($r_0$), momentum vector ($k_0$), spatial spread ($\sigma$), and mass.
*   **Multiple Solvers:** Choose between different numerical integration methods (Crank-Nicolson, SFFM) to calculate the wave's evolution.
*   **High-Performance 3D Rendering:** View the probability density mapped to height (Z-axis) and the complex quantum phase mapped to color, fully cached for lag-free playback.

## How do I start it?
This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management. 

1. **Prerequisites:** Ensure you have Python >= 3.14 and `uv` installed on your system.
2. **Install Dependencies:** Navigate to the project root (where `pyproject.toml` is located) and synchronize the environment:
   ```bash
   uv sync
   ```
3. **Run the Application:**
    ```bash
    uv run main.py
    ```

## How do I use it?
*   **Setting up the Physics:** Click the **Simulation Setup** button. Here you can:
    *   Select your numerical solver.
    *   Choose a preset potential or select "Brush Potential" to draw a custom potential directly onto the canvas. 
    *   Select "Set Wavepacket" to click-and-drag on the canvas. The dot represents the starting position ($r_0$) and the red line represents the momentum vector ($k_0$).
    *   Adjust the initial parameters: covariant matrix elements $s_{xx}$, $s_{yy}$, $s_{xy}$ and particle mass.
    *   Click **Save & Update Simulation** to calculate the wave evolution.
*   **Playback Controls:** Use the **Play** and **Pause** buttons to watch the animation, or drag the timeline slider to manually scrub forward and backward through time.
*   **Camera Controls:** Left-click and drag on the 3D viewport to rotate the camera. Scroll your mouse wheel to zoom in and out.
*   **Customizing the View:** Click the **Settings** button to adjust the playback framerate, the total number of frames, the grid resolution, and the visual amplitude scaling of the wave and potential meshes.
