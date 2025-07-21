#!/bin/bash

source /opt/ros/galactic/setup.bash
source /aicv-sensor-drivers/install/local_setup.bash

UTC_DATETIME=$(date -u +"%Y-%m-%d_%H-%M-%S")

# intrinsic calibration
echo -e "\n\n<--- Collect data for Camera Intrinsic calibration --->"
mkdir -p /mnt/$UTC_DATETIME/data/intrinsics_$UTC_DATETIME
cd /mnt/$UTC_DATETIME/data/intrinsics_$UTC_DATETIME
echo "Data collection duration: 120 seconds"
read -p "Press Enter to continue?" ENTER
echo "Data collection will begin in 5 seconds.."
sleep 5
ros2 bag record -d 120 -o intrinsics_$UTC_DATETIME $RGB_IMAGE_TOPIC &
INTRINSICPID=$!
count=120
while [ $count -gt 0 ]
do
  echo -ne "Countdown: $count s\r"
  count=$((count-1))
  sleep 1
done

kill $INTRINSICPID && sleep 5

# Ask if user wants to keep the file
echo -e "\n\n##### recording complete! Your results are in /mnt/$UTC_DATETIME #####"
read -p "Do you want to keep this rosbag? [y/N]: " keep

if [[ "$keep" =~ ^[Yy]$ ]]; then
    echo "Rosbag saved"
    exit 0
else
    echo "Deleting rosbag..."
    rm -rf "/mnt/$UTC_DATETIME"
    echo "Rosbag deleted."
    exit 0
fi
