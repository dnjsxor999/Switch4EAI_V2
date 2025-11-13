#!/bin/bash

set -e

echo "=== Starting Virtual Camera ==="

# Cleanup
echo "Cleaning up old processes..."
sudo killall -9 ffmpeg gst-launch-1.0 2>/dev/null || true
sleep 1

# Reset physical camera
echo "Resetting physical camera..."
sudo modprobe -r uvcvideo
sleep 1
sudo modprobe uvcvideo
sleep 2

# Configure virtual camera
echo "Configuring virtual camera..."
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="Virtual Camera" exclusive_caps=1
sleep 2

# Verify devices
if [ ! -e /dev/video0 ]; then
    echo "Error: /dev/video0 does not exist"
    exit 1
fi

if [ ! -e /dev/video2 ]; then
    echo "Error: /dev/video2 does not exist"
    exit 1
fi

echo "✓ Devices ready"
echo ""

# Attempt to start streaming
echo "Starting stream..."
echo "Source: /dev/video0"
echo "Target: /dev/video2"
echo "Press Ctrl+C to stop"
echo ""

# Try different methods
if timeout 2s ffmpeg -loglevel quiet -f v4l2 -i /dev/video0 -pix_fmt yuv420p -f v4l2 /dev/video2 2>/dev/null; then
    echo "Using method: Simple conversion"
    ffmpeg -f v4l2 -i /dev/video0 -pix_fmt yuv420p -f v4l2 /dev/video2
elif timeout 2s ffmpeg -loglevel quiet -f v4l2 -i /dev/video0 -pix_fmt bgr0 -f v4l2 /dev/video2 2>/dev/null; then
    echo "Using method: BGR0 format"
    ffmpeg -f v4l2 -i /dev/video0 -pix_fmt bgr0 -f v4l2 /dev/video2
else
    echo "Using method: Full parameters"
    ffmpeg -f v4l2 -input_format yuyv422 -video_size 2560x1600 -framerate 30 -i /dev/video0 \
      -vf "scale=1280:720,format=yuv420p" \
      -pix_fmt yuv420p \
      -f v4l2 /dev/video2
fi
