#!/bin/bash
# arrange-windows.sh - Auto arrange 5 windows for 2560x1440 resolution
# Author: Your Name
# Date: 2024

# Screen resolution
SCREEN_WIDTH=2560
SCREEN_HEIGHT=1440

# Window dimensions
LEFT_WIDTH=1280
RIGHT_WIDTH=1280
LEFT_HEIGHT=480
RIGHT_TOP_HEIGHT=720
RIGHT_BOTTOM_HEIGHT=720

echo "=========================================="
echo "Starting Window Arrangement (2560x1440)"
echo "=========================================="

# Display current windows
echo "Current windows:"
wmctrl -l
echo "=========================================="

# Get MuJoCo windows (3 windows)
MUJOCO_WINS=($(wmctrl -l | grep "MuJoCo : g1_mocap" | awk '{print $1}'))

echo "Found ${#MUJOCO_WINS[@]} MuJoCo windows"

# Arrange MuJoCo windows (left column, equal size)
for i in {0..2}; do
    if [ $i -lt ${#MUJOCO_WINS[@]} ]; then
        WIN_ID=${MUJOCO_WINS[$i]}
        Y_POS=$((i * LEFT_HEIGHT))
        
        echo "Arranging MuJoCo window $((i+1)): $WIN_ID"
        
        # Remove all window states
        wmctrl -i -r $WIN_ID -b remove,maximized_vert,maximized_horz,fullscreen,hidden,shaded
        sleep 0.1
        
        # Move window
        wmctrl -i -r $WIN_ID -e 0,0,$Y_POS,$LEFT_WIDTH,$LEFT_HEIGHT
        sleep 0.1
        
        # Set again to ensure it takes effect
        wmctrl -i -r $WIN_ID -e 0,0,$Y_POS,$LEFT_WIDTH,$LEFT_HEIGHT
        
        echo "✓ MuJoCo $((i+1)) -> (0, $Y_POS, ${LEFT_WIDTH}x${LEFT_HEIGHT})"
    fi
done

# Get video window
VIDEO_WIN=$(wmctrl -l | grep "/dev/video2" | awk '{print $1}')

if [ ! -z "$VIDEO_WIN" ]; then
    echo ""
    echo "Arranging Video window: $VIDEO_WIN"
    
    # Remove all states
    wmctrl -i -r $VIDEO_WIN -b remove,maximized_vert,maximized_horz,fullscreen,hidden,shaded
    sleep 0.2
    
    # Move window (execute twice to ensure it works)
    wmctrl -i -r $VIDEO_WIN -e 0,$LEFT_WIDTH,0,$RIGHT_WIDTH,$RIGHT_TOP_HEIGHT
    sleep 0.1
    wmctrl -i -r $VIDEO_WIN -e 0,$LEFT_WIDTH,0,$RIGHT_WIDTH,$RIGHT_TOP_HEIGHT
    
    echo "✓ Video -> ($LEFT_WIDTH, 0, ${RIGHT_WIDTH}x${RIGHT_TOP_HEIGHT})"
else
    echo "✗ Video window not found"
fi

# Get Terminal window
TERM_WIN=$(wmctrl -l | grep "luyd@pnd-System-Product-Name: ~" | awk '{print $1}')

if [ ! -z "$TERM_WIN" ]; then
    echo ""
    echo "Arranging Terminal window: $TERM_WIN"
    
    wmctrl -i -r $TERM_WIN -b remove,maximized_vert,maximized_horz,fullscreen,hidden,shaded
    sleep 0.2
    
    wmctrl -i -r $TERM_WIN -e 0,$LEFT_WIDTH,$RIGHT_TOP_HEIGHT,$RIGHT_WIDTH,$RIGHT_BOTTOM_HEIGHT
    sleep 0.1
    wmctrl -i -r $TERM_WIN -e 0,$LEFT_WIDTH,$RIGHT_TOP_HEIGHT,$RIGHT_WIDTH,$RIGHT_BOTTOM_HEIGHT
    
    echo "✓ Terminal -> ($LEFT_WIDTH, $RIGHT_TOP_HEIGHT, ${RIGHT_WIDTH}x${RIGHT_BOTTOM_HEIGHT})"
fi

echo "=========================================="
echo "Done!"
echo "=========================================="

# Final layout summary
echo ""
echo "Final Layout:"
echo "Left Top:    MuJoCo 1    (0, 0, 1280x480)"
echo "Left Middle: MuJoCo 2    (0, 480, 1280x480)"  
echo "Left Bottom: MuJoCo 3    (0, 960, 1280x480)"
echo "Right Top:   Video       (1280, 0, 1280x720)"
echo "Right Bottom: Terminal   (1280, 720, 1280x720)"

# Desktop notification
command -v notify-send >/dev/null 2>&1 && \
    notify-send "Window Arrangement" "5 windows arranged successfully\nResolution: 2560x1440" -i preferences-desktop-display