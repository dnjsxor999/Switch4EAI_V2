# Output Interpolation Module

## Overview

The interpolation module increases the output frequency of the pipeline by interpolating multiple frames between consecutive pipeline outputs. You can configure how many interpolated frames to generate, allowing flexible output rates.

## How It Works

### Timeline Examples

**With 1 interpolation (default, 10Hz)**:
```
t=0.0s: Start processing frame_0
t=0.2s: frame_0 processed → result_0
        Update interpolator (f_prev=None, f_curr=result_0)
        No output sent (initial constant lag)

t=0.4s: frame_1 processed → result_1
        Update interpolator (f_prev=result_0, f_curr=result_1)
        Send result_0 immediately
        Schedule interpolated for t=0.5s (alpha=0.5)

t=0.5s: Send interpolated(result_0, result_1, alpha=0.5)

t=0.6s: frame_2 processed → result_2
        Update interpolator (f_prev=result_1, f_curr=result_2)
        Send result_1 immediately
        Schedule interpolated for t=0.7s (alpha=0.5)

t=0.7s: Send interpolated(result_1, result_2, alpha=0.5)

...continues with 5Hz actual + 5Hz interpolated = 10Hz total
```

**With 3 interpolations (20Hz)**:
```
t=0.0s: Start processing frame_0
t=0.2s: frame_0 processed → result_0
        No output sent (initial lag)

t=0.4s: frame_1 processed → result_1
        Send result_0 immediately
        Schedule 3 interpolations at t=0.45, 0.50, 0.55 (alpha=0.25, 0.5, 0.75)

t=0.45s: Send interpolated(result_0, result_1, alpha=0.25)
t=0.50s: Send interpolated(result_0, result_1, alpha=0.50)
t=0.55s: Send interpolated(result_0, result_1, alpha=0.75)

t=0.6s: frame_2 processed → result_2
        Send result_1 immediately
        Schedule 3 interpolations at t=0.65, 0.70, 0.75 (alpha=0.25, 0.5, 0.75)

t=0.65s: Send interpolated(result_1, result_2, alpha=0.25)
t=0.70s: Send interpolated(result_1, result_2, alpha=0.50)
t=0.75s: Send interpolated(result_1, result_2, alpha=0.75)

...continues with 5Hz actual + 15Hz interpolated = 20Hz total
```

### Key Insight

**Output Frequency Formula**:
```
output_hz = xHz × (num_interpolations + 1)

Examples (if original rate is 5Hz):
- num_interpolations = 1 → 10Hz
- num_interpolations = 2 → 15Hz
- num_interpolations = 3 → 20Hz
- num_interpolations = 4 → 25Hz
```

### Key Features

1. **Flexible Frequency**: Configure output rate from 10Hz to 30Hz+ via `--num-interp`
2. **Constant Initial Lag**: One-time 0.2s lag at startup (acceptable)
3. **No Accumulating Lag**: Real-time operation preserved
4. **Smooth Interpolation**: 
   - Linear interpolation (LERP) for positions and velocities
   - Spherical linear interpolation (SLERP) for rotations
5. **Background Thread**: Interpolated outputs sent at precisely the right times

## Usage

### Basic Usage (Single-threaded)

```bash
# With 1 interpolation (default, 10Hz if original rate is 5Hz)
python scripts/run_stream_to_robot.py --camera 0

# With 2 interpolations (15Hz if original rate is 5Hz)
python scripts/run_stream_to_robot.py --camera 0 --num-interp 2

# With 3 interpolations (20Hz if original rate is 5Hz)
python scripts/run_stream_to_robot.py --camera 0 --num-interp 3

# Without interpolation (5Hz, original behavior)
python scripts/run_stream_to_robot.py --camera 0 --no-interpolation
```


### Video Input

```bash
python scripts/run_stream_to_robot.py --video path/to/video.mp4 --num-interp 2
```

### Debug Timing Mode

To verify that interpolation is working correctly and outputs are being sent at the expected intervals:

```bash
# Debug with 1 interpolation (10Hz, ~100ms intervals)
python scripts/run_stream_to_robot.py --camera 0 --debug-timing

# Debug with 3 interpolations (20Hz, ~50ms intervals)
python scripts/run_stream_to_robot.py --camera 0 --num-interp 3 --debug-timing

# Expected output example (num_interpolations=3):
# [DEBUG #0001] ACTUAL | First output (no interval)
# [DEBUG #0002] ACTUAL | Δt =  200.2ms | Target: ~50ms
# [DEBUG #0003] INTERP | Δt =   50.1ms | Target: ~50ms
# [DEBUG #0004] INTERP | Δt =   49.8ms | Target: ~50ms
# [DEBUG #0005] INTERP | Δt =   50.2ms | Target: ~50ms
# [DEBUG #0006] ACTUAL | Δt =   49.9ms | Target: ~50ms
# [DEBUG #0007] INTERP | Δt =   50.0ms | Target: ~50ms
```

This mode prints timing information for every UDP output:
- **ACTUAL**: Real processing result from pipeline
- **INTERP**: Interpolated output between two actuals
- **Δt**: Time interval since last output
- **Target**: Expected interval based on num_interpolations

## Command-Line Arguments

- `--camera INDEX`: Camera index to use (e.g., 0, 1, 2)
- `--video PATH`: Path to video file instead of camera
- `--list-cams`: List available cameras and exit
- `--no-interpolation`: Disable interpolation (run at native ~5Hz)
- `--num-interp N`: Number of interpolated frames between actuals (default: 1)
  - `1` → 10Hz (5Hz actual + 5Hz interpolated)
  - `2` → 15Hz (5Hz actual + 10Hz interpolated)
  - `3` → 20Hz (5Hz actual + 15Hz interpolated)
  - `4` → 25Hz (5Hz actual + 20Hz interpolated)
  - etc.
- `--debug-timing`: Print timing information for each UDP output

## UDP Output Format

All UDP messages include an `"interpolated"` boolean field:

### Visualize Mode (qpos)
```json
{
  "type": "qpos",
  "qpos": [...],
  "root_pos": [...],
  "root_rot_xyzw": [...],
  "dof_pos": [...],
  "root_vel": [...],
  "root_ang_vel": [...],
  "dof_vel": [...],
  "interpolated": false
}
```

### Headless Mode (motion_data)
```json
{
  "type": "motion_data",
  "fps": 30,
  "root_pos": [...],
  "root_rot": [...],
  "dof_pos": [...],
  "local_body_pos": [...],
  "root_vel": [...],
  "root_ang_vel": [...],
  "dof_vel": [...],
  "interpolated": true
}
```

## Implementation Details

### Interpolation Methods

For `num_interp = N`, alpha values are calculated as:
```python
alphas = [(i + 1) / (N + 1) for i in range(N)]

Examples:
- N=1: [0.5]
- N=2: [0.333, 0.667]
- N=3: [0.25, 0.5, 0.75]
- N=4: [0.2, 0.4, 0.6, 0.8]
```

1. **Position Interpolation (LERP)**:
   ```
   pos_interp = (1 - α) * pos_prev + α * pos_curr
   ```

2. **Rotation Interpolation (SLERP)**:
   - Uses scipy's `Slerp` for smooth quaternion interpolation
   - Handles both wxyz and xyzw quaternion formats

3. **Velocity Interpolation (LERP)**:
   - Velocities are also linearly interpolated
   - Ensures smooth transitions

### Thread Safety

- Main thread: Processes frames and updates interpolator
- Background thread: Monitors interpolator and sends interpolated outputs
- Thread-safe by design: Main thread only writes, background thread only reads

### Performance

- Background thread checks every 10ms for pending interpolated outputs
- Minimal overhead (~0.1% CPU per interpolation)
- No impact on processing speed
- Memory: Stores only 2 outputs at a time (f_prev, f_curr)

## Performance Comparison

| num_interpolations | Output Hz | Interval (ms) | Use Case |
|--------------------|-----------|---------------|----------|
| 0 (disabled)       | 5Hz       | 200ms         | Low bandwidth |
| 1 (default)        | 10Hz      | 100ms         | Standard use |
| 2                  | 15Hz      | 67ms          | Smoother motion |
| 3                  | 20Hz      | 50ms          | High-frequency control |
| 4                  | 25Hz      | 40ms          | Ultra-smooth |

**Recommendation**: Start with `num_interp=1` (~10Hz) and increase if needed.

## Troubleshooting

### Issue: Output not at expected frequency
- Check `--debug-timing` output
- Verify `--no-interpolation` flag is NOT set
- Ensure background thread started (look for startup message)

### Issue: Jittery motion
- System may be overloaded with too many interpolations
- Try reducing `--num-interp`
- Check CPU usage

### Issue: Accumulated lag
- This should NOT happen - interpolation preserves real-time operation
- If observed, report as a bug

## Future Improvements

Potential enhancements:
- Adaptive num_interpolations based on system load
- Higher-order interpolation (cubic, spline)
- Predictive interpolation using motion history
- Per-joint interpolation quality settings
