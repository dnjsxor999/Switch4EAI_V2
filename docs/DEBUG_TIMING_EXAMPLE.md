# Debug Timing Mode - Usage Example

## Purpose

The `--debug-timing` flag helps you verify that:
1. Interpolation is working correctly
2. Outputs are being sent at the expected ~100ms intervals (10Hz)
3. ACTUAL and INTERP outputs are alternating properly

## Usage

```bash
# Single-threaded version
python scripts/run_stream_to_robot.py --camera 0 --debug-timing

# Multi-threaded version
python scripts/run_stream_to_robot_mt.py --camera 0 --debug-timing

# With video input
python scripts/run_stream_to_robot.py --video path/to/video.mp4 --debug-timing

# Without interpolation (to compare)
python scripts/run_stream_to_robot.py --camera 0 --no-interpolation --debug-timing
```

## Expected Output

### With Interpolation (10Hz)

```
======================================================================
DEBUG TIMING MODE ENABLED
======================================================================
Legend:
  ACTUAL = Real processing result from pipeline
  INTERP = Interpolated output between two actuals
  Δt     = Time interval since last output
  Target = Expected interval (~100ms for 10Hz)
======================================================================
Interpolation enabled (alpha=0.5)
Pipeline started...

[DEBUG #0001] ACTUAL | First output (no interval)
[DEBUG #0002] ACTUAL | Δt =  201.2ms | Target: ~100ms
[DEBUG #0003] INTERP | Δt =  100.6ms | Target: ~100ms
[DEBUG #0004] ACTUAL | Δt =  100.8ms | Target: ~100ms
[DEBUG #0005] INTERP | Δt =   99.5ms | Target: ~100ms
[DEBUG #0006] ACTUAL | Δt =  100.3ms | Target: ~100ms
[DEBUG #0007] INTERP | Δt =  100.1ms | Target: ~100ms
[DEBUG #0008] ACTUAL | Δt =   99.8ms | Target: ~100ms
[DEBUG #0009] INTERP | Δt =  100.2ms | Target: ~100ms
[DEBUG #0010] ACTUAL | Δt =  100.0ms | Target: ~100ms
...
```

**Analysis**:
- First output has no interval (startup)
- Second output shows ~200ms (one full processing cycle)
- After that, outputs alternate between ACTUAL and INTERP
- Intervals are consistently ~100ms (10Hz total)
- Pattern: ACTUAL → INTERP → ACTUAL → INTERP ...

### Without Interpolation (5Hz)

```
======================================================================
DEBUG TIMING MODE ENABLED
======================================================================
Legend:
  ACTUAL = Real processing result from pipeline
  INTERP = Interpolated output between two actuals
  Δt     = Time interval since last output
  Target = Expected interval (~100ms for 10Hz)
======================================================================
Running at native ~5Hz (no interpolation)
Pipeline started...

[DEBUG #0001] ACTUAL | First output (no interval)
[DEBUG #0002] ACTUAL | Δt =  201.5ms | Target: ~100ms
[DEBUG #0003] ACTUAL | Δt =  200.3ms | Target: ~100ms
[DEBUG #0004] ACTUAL | Δt =  201.1ms | Target: ~100ms
[DEBUG #0005] ACTUAL | Δt =  199.8ms | Target: ~100ms
[DEBUG #0006] ACTUAL | Δt =  200.5ms | Target: ~100ms
[DEBUG #0007] ACTUAL | Δt =  200.2ms | Target: ~100ms
[DEBUG #0008] ACTUAL | Δt =  200.0ms | Target: ~100ms
...
```

**Analysis**:
- All outputs are ACTUAL (no interpolation)
- Intervals are consistently ~200ms (5Hz)
- No INTERP outputs

## Interpreting the Results

### Good Signs ✓

1. **With Interpolation**:
   - ACTUAL and INTERP alternate consistently
   - Intervals are ~100ms (±10ms is normal)
   - Total output rate is ~10Hz

2. **Without Interpolation**:
   - All ACTUAL outputs
   - Intervals are ~200ms (processing time per frame)
   - Total output rate is ~5Hz

### Potential Issues ⚠️

1. **Irregular Intervals**:
   ```
   [DEBUG #0005] INTERP | Δt =  100.2ms | Target: ~100ms
   [DEBUG #0006] ACTUAL | Δt =  350.5ms | Target: ~100ms  ← Too long!
   [DEBUG #0007] INTERP | Δt =   15.2ms | Target: ~100ms  ← Too short!
   ```
   - Could indicate system load issues
   - Check CPU usage
   - Verify no other heavy processes running

2. **Missing INTERP Outputs**:
   ```
   [DEBUG #0005] ACTUAL | Δt =  200.2ms | Target: ~100ms
   [DEBUG #0006] ACTUAL | Δt =  200.5ms | Target: ~100ms
   [DEBUG #0007] ACTUAL | Δt =  199.8ms | Target: ~100ms
   ```
   - Interpolation may not be enabled
   - Check that `--no-interpolation` is NOT set
   - Verify background thread started

3. **Only INTERP Outputs** (very rare):
   ```
   [DEBUG #0005] INTERP | Δt =  100.2ms | Target: ~100ms
   [DEBUG #0006] INTERP | Δt =  100.5ms | Target: ~100ms
   [DEBUG #0007] INTERP | Δt =   99.8ms | Target: ~100ms
   ```
   - Pipeline may not be processing new frames
   - Check camera connection
   - Verify video input is not paused

## Real-World Example with Variance

Actual output will have some natural variance:

```
[DEBUG #0001] ACTUAL | First output (no interval)
[DEBUG #0002] ACTUAL | Δt =  203.1ms | Target: ~100ms
[DEBUG #0003] INTERP | Δt =  101.6ms | Target: ~100ms
[DEBUG #0004] ACTUAL | Δt =  101.5ms | Target: ~100ms
[DEBUG #0005] INTERP | Δt =   98.4ms | Target: ~100ms
[DEBUG #0006] ACTUAL | Δt =  102.1ms | Target: ~100ms
[DEBUG #0007] INTERP | Δt =   99.8ms | Target: ~100ms
[DEBUG #0008] ACTUAL | Δt =   97.9ms | Target: ~100ms
[DEBUG #0009] INTERP | Δt =  103.2ms | Target: ~100ms
[DEBUG #0010] ACTUAL | Δt =   98.7ms | Target: ~100ms
[DEBUG #0011] INTERP | Δt =  100.5ms | Target: ~100ms
```

**This is normal**: ±5ms variance is expected due to:
- OS scheduling jitter
- Background processes
- Camera frame timing variations
- Network latency (if UDP receiver on different machine)

**Average over 10 outputs** (excluding first two):
- Mean: 100.2ms
- Frequency: ~9.98Hz ≈ 10Hz ✓

## Performance Metrics to Track

When debugging, track these metrics:

1. **Average Interval**:
   - With interpolation: Should be ~100ms (10Hz)
   - Without interpolation: Should be ~200ms (5Hz)

2. **Standard Deviation**:
   - Good: < 10ms
   - Acceptable: < 20ms
   - Poor: > 20ms (investigate system issues)

3. **Output Pattern**:
   - With interpolation: ACTUAL, INTERP, ACTUAL, INTERP, ...
   - Without interpolation: ACTUAL, ACTUAL, ACTUAL, ...

4. **Dropped Frames**:
   - Large gaps (> 300ms) indicate processing delays
   - Check system load and optimize if needed

## Tips

1. **Start with debug mode**: Always test with `--debug-timing` first to verify interpolation is working
2. **Compare modes**: Run with and without `--no-interpolation` to see the difference
3. **Watch for patterns**: Look for consistent ACTUAL/INTERP alternation
4. **Check averages**: Short-term variance is normal, but average should be ~100ms
5. **System load**: If intervals are irregular, close other applications

## Disabling Debug Mode

Simply remove the `--debug-timing` flag:

```bash
# Normal operation (no debug output)
python scripts/run_stream_to_robot.py --camera 0
```

The interpolation will still work, you just won't see the timing printouts.

