# Quaint 🌌

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.14%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-purple)](https://docs.astral.sh/uv/)

> A Python-based desktop application for real-time, interactive 3D visualization of the time-dependent 2D Schrödinger equation.

<img src="look.gif" width="500">

## 📖 About

**Quaint** provides a real-time, interactive environment to explore quantum mechanics. Built with **PyQt6**, **PyQtGraph**, and **SciPy**, it specifically focuses on visualizing how wave packets interact with various potential energy landscapes.

## ✨ Features

* **Custom Potentials:** Use a built-in canvas to brush or erase custom potential energy barriers, or load mathematical presets (e.g., Harmonic Oscillator, Gaussian Bump).
* **Initial Conditions:** Visually set the wave packet's starting position ($r_0$), momentum vector ($k_0$), spatial spread ($\sigma$), and mass.
* **Multiple Solvers:** Choose between different numerical integration methods (e.g., Crank-Nicolson, SFFM) to calculate the wave's evolution.
* **High-Performance 3D Rendering:** View the probability density mapped to height (Z-axis) and the complex quantum phase mapped to color, fully cached for lag-free playback.

## 🚀 Installation

This project uses [uv](https://docs.astral.sh/uv/) for incredibly fast dependency and environment management.

**1. Prerequisites** Ensure you have Python >= 3.14 and uv installed on your system.

**2. Clone the repository**
```
git clone [https://github.com/Hansus00/Quaint.git](https://github.com/Hansus00/Quaint.git)
cd Quaint
```

**3. Install Dependencies** Synchronize the environment using uv:
```uv sync```

**4. Run the Application**
```uv run main.py```

## 🎮 Usage

### ⚙️ Setting up the Physics
Click the **Simulation Setup** button to customize your experiment:
* **Numerical Solver:** Select your preferred integration method.
* **Potential Energy:** Choose a preset potential or select **"Brush Potential"** to draw a custom landscape directly onto the canvas.
* **Wavepacket Configuration:** Select **"Set Wavepacket"** to click-and-drag on the canvas.
  * The **dot** represents the starting position ($r_0$).
  * The **red line** represents the momentum vector ($k_0$).
* **Initial Parameters:** Adjust covariant matrix elements ($s_{xx}$, $s_{yy}$, $s_{xy}$) and particle mass.
* Click **Save & Update Simulation** to compute the wave's evolution over time.

### ⏯️ Playback Controls
* Use the **Play** and **Pause** buttons to watch the animation.
* Drag the **timeline slider** to manually scrub forward and backward through time.

### 🎥 Camera Controls & View
* **Rotate:** Left-click and drag on the 3D viewport.
* **Zoom:** Scroll your mouse wheel in and out.
* **Settings:** Click the **Settings** button to adjust playback framerate, total number of frames, grid resolution, and visual amplitude scaling of the wave and potential meshes.

## 📄 License

This project is licensed under the GPL-3.0 License. See the LICENSE file for more details.