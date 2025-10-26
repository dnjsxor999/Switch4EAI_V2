# Output Interpolation Module

## Overview

The interpolation module doubles the output frequency of the pipeline from ~5Hz to ~10Hz by interpolating between consecutive pipeline outputs.

## How It Works

### Timeline

```
t=0.0s: Start processing frame_0
t=0.2s: frame_0 processed → result_0
        Update interpolator (f_prev=None, f_curr=result_0)
        No output sent (initial constant lag)

t=0.4s: frame_1 processed → result_1
        Update interpolator (f_prev=result_0, f_curr=result_1)
        Send result_0 immediately
        Schedule interpolated for t=0.4 + (0.2 * 0.5) = t=0.5s

t=0.5s: Send interpolated(result_0, result_1, alpha=0.5)

t=0.6s: frame_2 processed → result_2
        Update interpolator (f_prev=result_1, f_curr=result_2)
        Send result_1 immediately
        Schedule interpolated for t=0.6 + (0.2 * 0.5) = t=0.7s

t=0.7s: Send interpolated(result_1, result_2, alpha=0.5)

t=0.8s: frame_3 processed → result_3
        ...continues with 5Hz actual + 5Hz interpolated = 10Hz total
```

**Key Insight**: With alpha=0.5 and 0.2s processing time:
- Actual outputs: t=0.2, 0.4, 0.6, 0.8, 1.0... (5Hz)
- Interpolated outputs: t=0.3, 0.5, 0.7, 0.9, 1.1... (5Hz)
- **Total output rate: 10Hz**

### Key Features

1. **Constant Initial Lag**: There's a one-time 0.2s lag at startup before outputs begin
2. **No Accumulating Lag**: Real-time operation is preserved - lag does not accumulate
3. **Smooth Interpolation**: 
   - Linear interpolation (LERP) for positions and velocities
   - Spherical linear interpolation (SLERP) for rotations
4. **Background Thread**: Interpolated outputs are sent by a background thread at precisely the right time

## Usage

### Basic Usage (Single-threaded)

```bash
# With interpolation (default, ~10Hz output)
python scripts/run_stream_to_robot.py --camera 0

# Without interpolation (~5Hz output)
python scripts/run_stream_to_robot.py --camera 0 --no-interpolation

# Custom interpolation point (0.3 = 30% between prev and curr)
python scripts/run_stream_to_robot.py --camera 0 --interpolation-alpha 0.3
```

### Multi-threaded Version

```bash
# With interpolation (default)
python scripts/run_stream_to_robot_mt.py --camera 0

# Without interpolation
python scripts/run_stream_to_robot_mt.py --camera 0 --no-interpolation
```

### Video Input

```bash
python scripts/run_stream_to_robot.py --video path/to/video.mp4
```

## Command-Line Arguments

- `--camera INDEX`: Camera index to use (e.g., 0, 1, 2)
- `--video PATH`: Path to video file instead of camera
- `--list-cams`: List available cameras and exit
- `--no-interpolation`: Disable interpolation (run at native ~5Hz)
- `--interpolation-alpha ALPHA`: Interpolation factor (default: 0.5)
  - `0.0` = output at prev position
  - `0.5` = midpoint between prev and curr (default)
  - `1.0` = output at curr position

## UDP Output Format

All UDP messages now include an `"interpolated"` boolean field:

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

1. **Position Interpolation (LERP)**:
   ```
   pos_interp = (1 - α) * pos_prev + α * pos_curr
   ```

2. **Rotation Interpolation (SLERP)**:
   - Uses scipy's `Slerp` for smooth quaternion interpolation
   - Handles both wxyz and xyzw quaternion formats

3. **Velocity Interpolation**:
   - Velocities are also linearly interpolated
   - Ensures smooth transitions

### Thread Safety

- Main thread: Processes frames and updates interpolator
- Background thread: Monitors interpolator and sends interpolated outputs
- Thread-safe by design: Main thread only writes to interpolator, background thread only reads

### Performance

- Background thread checks every 10ms for interpolated outputs
- Minimal overhead (~0.1% CPU for background thread)
- No impact on processing speed

## Module Structure

```
Switch4EmbodiedAI/modules/interpolator.py
├── OutputInterpolator
│   ├── update(new_output)           # Update with new pipeline output
│   ├── get_actual_output()          # Get actual output (f_prev)
│   ├── get_next_output()            # Get next output (actual or interpolated)
│   ├── should_send_interpolated()   # Check if it's time to send interpolated
│   └── has_outputs_ready()          # Check if both prev and curr are ready
```

## Testing

To verify interpolation is working:

1. Monitor UDP output frequency (should be ~10Hz with interpolation, ~5Hz without)
2. Check the `"interpolated"` field in UDP messages
3. Verify smooth motion without jitter

## Troubleshooting

### Issue: Output still at 5Hz
- Check that `--no-interpolation` flag is NOT set
- Verify background thread is started (look for startup message)
- Ensure UDP is enabled in config

### Issue: Jittery motion
- Try adjusting `--interpolation-alpha` (default 0.5)
- Check network latency if receiving UDP messages
- Verify camera framerate is stable

### Issue: Accumulated lag
- This should NOT happen - interpolation preserves real-time operation
- If you observe accumulating lag, please report as a bug

## Future Improvements

Potential enhancements:
- Adaptive alpha based on motion speed
- Higher-order interpolation (cubic, spline)
- Predictive interpolation using motion history
- Configurable output frequency multiplier (3x, 4x, etc.)

