# Quaint — Agent Guide

2D Schrödinger equation simulator with PyQt6 UI and 3D visualization (|ψ|² → height, phase → color).

## Commands

```bash
uv sync && uv run main.py
uvx ty check frontend/          # typecheck UI
uv run test/universal_tester.py --config test/cfg/infiniteWell.json --out /tmp/out
```

- Entry: `main.py` → `frontend.main_window.MainWindow`
- `pyproject.toml` name is `zps`; Python `>=3.10.11` (ignore README’s 3.14)
- Stack: PyQt6, pyqtgraph, numpy, scipy; physics in `backend/`, UI in `frontend/`

## Layout

| Path | Role |
|------|------|
| `backend/Potential.py` | V(x,y) grids + presets |
| `backend/StationaryWaveFunc.py` | ψ; `GaussianPacket` for ψ₀ |
| `backend/Solver.py` | CN, SSFM, SSFMSymmetric, Constant |
| `backend/Params.py` | JSON config (`WellType`, `SolverType`) |
| `frontend/main_window.py` | Orchestrator |
| `frontend/setup_drawer.py` | Setup dialog + canvas |
| `frontend/simulation_thread.py` | Precomputes all frames |
| `frontend/animation_widget.py` | OpenGL meshes |
| `test/universal_tester.py` | Headless CLI tests |

**Flow:** `SetupDrawer` → `MainWindow.apply_setup` → `SimulationThread` (N×`steps_per_frame` solver steps) → `AnimationWidget`. `Settings` is visual-only; physics changes need **Save & Update Simulation**.

## Backend

- Grids: `(Nx, Ny)`, `indexing="ij"`. Flatten: C-order `(i,j) → i*Ny + j`.
- **Solvers in UI:** Crank-Nicolson, SSFM, Constant. `SSFMSymmetric` exists but is test-only.
- `Params`: `updates_max` = frame count; `delta_n` = steps/frame (UI: `steps_per_frame`).

## Frontend pitfalls

**Y-axis flip (Qt top-left vs physics):** `matrix[:, ::-1]` between canvas and solver; momentum `kx = (tip_x - r0_x) * 0.1`, `ky = -(tip_y - r0_y) * 0.1`. Break this → UI/tests diverge. Search `::-1` and `1.0 - (y / height)` in `setup_drawer.py` / `main_window.py`.

**Qt + ty:** Event overrides use param `a0` with `| None` types; check `QPoint | None` with `is not None`, not `if point:`.

**Memory:** optional `psutil` caps frames (`SetupDrawer`) and GL cache (`MainWindow`).

## Gaps

- `SSFMSymmetric` / `SYM_SSFM` not wired in GUI
- Params JSON stores `well_type`, not full custom potential matrix
- `delta_r` FIXME in solvers (default 1)

## Where to edit

| Task | Files |
|------|-------|
| New potential | `Potential.py`, `setup_drawer.load_preset_potential` |
| New solver | `Solver.py`, `main_window.switch_simulation_method`, `Params`, setup combo |
| Coordinates / ψ₀ | `setup_drawer`, `main_window.apply_setup`, `GaussianPacket` |
| Rendering perf | `animation_widget.py` |

User docs: [README.md](README.md).
