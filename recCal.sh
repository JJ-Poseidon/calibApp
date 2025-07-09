#!/bin/bash

# Prompt user to begin
echo "Press ENTER to begin calibration recording..."
read

# Countdown before starting
echo "Calibration begins in:"
for i in {5..1}; do
    echo "$i..."
    sleep 1
done

# Generate UTC timestamp for filename
utc_time=$(date -u +"%Y-%m-%dT_%H-%M-%SZ")
bag_name="calibration_$utc_time"
bag_path="/mnt/$bag_name"

# Start ros2 bag recording
echo "Recording ros bag to: $bag_path"
# ros2 bag record /camera1/image_raw -o "$bag_path" &
ros2 bag record -d 120 -o intrinsics_$UTC_DATETIME $RGB_IMAGE_TOPIC &
bag_pid=$!

# Countdown timer (120 seconds)
for ((i=120; i>0; i--)); do
    printf "\rRecording... %3d seconds left" "$i"
    sleep 1
done
echo -e "\nRecording complete!"

# Stop ros2 bag recording
echo "Stopping ros2 bag..."
kill $bag_pid
sleep 2  # Allow cleanup

# Ask if user wants to keep the file
echo "Bag file: $bag_path"
read -p "Do you want to keep this rosbag? [y/N]: " keep

if [[ "$keep" =~ ^[Yy]$ ]]; then
    echo "Rosbag kept at: $bag_path"
else
    echo "Deleting rosbag..."
    rm -rf "$bag_path"
    echo "Rosbag deleted."
fi
