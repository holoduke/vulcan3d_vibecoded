# quake-like — build & run

## Build (recommended: build.ps1 wrapper)

`build.ps1` sources `vcvars64.bat` itself, configures CMake (Ninja) on first
run, and builds — works from a vanilla PowerShell with no dev env set up:

```powershell
cd C:\Users\gillis\dev\quake-like
./build.ps1 -Config Release       # add -Clean to wipe build/, -Test to run tests
```

It auto-reconfigures when the cached `CMAKE_BUILD_TYPE` differs from `-Config`,
so switching Debug<->Release is safe. New shader files under `shaders/` and new
sources added to `CMakeLists.txt` are picked up on the next run.

If the final link fails with `LNK1104: cannot open file 'quake_like.exe'`, a
previous instance is still running — kill it first:
`Get-Process quake_like -ErrorAction SilentlyContinue | Stop-Process -Force`.

## Build (manual cmake)

From "Developer PowerShell for VS 2022" (env pre-loaded), or after sourcing
`vcvars64.bat` manually:

```powershell
cd C:\Users\gillis\dev\quake-like
cmake -S . -B build              # only needed on fresh checkout or after adding files
cmake --build build --config Release --target quake_like
```

Build artifacts: `build\quake_like.exe`, `build\shaders\*.spv` (auto-compiled
from `shaders/*.{vert,frag,tesc,tese,comp}` via CMake glob — new shader files
get picked up next `cmake -S . -B build`).

## Run

```powershell
cd C:\Users\gillis\dev\quake-like     # cwd MUST be project root or all textures load as gray placeholders
.\build\quake_like.exe
```

Validation is OFF by default in Release builds (zero overhead). Only set
`QLIKE_VK_VALIDATION=1` to enable Vulkan validation layers for debugging.

## CLI flags

- `--frames N` — run N frames then exit (-1 = forever, default)
- `--screenshot PATH` — capture one frame to PPM and exit
- `--screenshot-after N` — wait N frames before the screenshot (default 5)
- `--log PATH` — log file path (default `qlike.log`)
- `--autodemo SECS` — synth walk+fire input for SECS, then exit

## Quick recipes

- Smoke test, exits cleanly after ~2 sec: `.\build\quake_like.exe --frames 120`
- Take a screenshot: `.\build\quake_like.exe --screenshot test.ppm --screenshot-after 60 --frames 90`
- Interactive normal launch: `.\build\quake_like.exe`

## Gotchas

- `--screenshot PATH` alone runs **forever** until the frame is captured; it
  does not imply `--frames`. Pair with `--frames N` (N > screenshot-after) for a
  one-shot capture that exits.
- Screenshots are PPM (P6). Convert to PNG for viewing (no `magick`/`python` on
  this box — use a .NET `System.Drawing` snippet or any PPM viewer).
- Player pose (`player_pos_*`, `player_yaw`, `player_pitch`), sun angle, and all
  render toggles persist in `qlike_settings.cfg` (next to the exe, in `build/`)
  and are **restored on launch** — they override the in-code spawn defaults.
  Edit that file (or delete it to reset) to control where you spawn for a test
  capture.
- Terrain has two renderers: mesh (CDLOD, drawn by `cube.frag`) and raymarched
  (`terrain_raymarch.frag`), toggled by `terrain_raymarch_enabled`. Inline-RT
  features wired into `cube.frag` (e.g. voxel-tower shadows on the ground) only
  show on the **mesh** terrain; the raymarch path has its own lighting.
