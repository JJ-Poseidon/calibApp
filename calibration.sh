#!/bin/bash

source /opt/ros/galactic/setup.bash
source /aicv-sensor-drivers/install/local_setup.bash

UTC_DATETIME=$(date -u +"%Y-%m-%d_%H-%M-%S") # 2019-12-23_10-53-59

# # launch IMU
# cd /aicv-sensor-drivers/src/sensor_calibrator
# if ! ros2 node list | grep -q smr_imu_node; then 
#     ros2 run imu_driver smr_imu --ros-args --params-file ./config/calib_imu.yaml > /dev/null 2>&1 & 
#     sleep 5 # FIXME: Could take more than 5 seconds to run the node

#     # check if node is running
#     if ros2 node list | grep -q smr_imu_node; then 
#         echo -e "\n\n<--- Launched Imu driver --->"
#     else
#         echo -e "\n\n<--- Imu driver failed to launch --->"
#         exit 1
#     fi
# else
#     echo -e "\n\n<--- Imu driver already running --->"
# fi

# launch camera
# DON'T launch the camera here because it's already being launched by aicv-percy
# cd /aicv-sensor-drivers/src/sensor_calibrator
# if ! ros2 node list | grep -q smr_camera_node; then
#     # run node
#     ros2 run camera_driver smr_camera --ros-args --params-file ./config/ros_cam_4k_default.yaml > /dev/null 2>&1 &
#     sleep 5 # FIXME: Could take more than 5 seconds to run the node

#     # check if node is running
#     if ros2 node list | grep -q smr_camera_node; then
#         echo -e "\n\n<--- Launched camera driver --->"
#     else
#         echo -e "\n\n<--- Camera driver failed to launch --->"
#         exit 1
#     fi
# else
#     echo -e "\n\n<--- Camera driver already running --->"
# fi

# intrinsic calibration
echo -e "\n\n<--- Collect data for Camera Intrinsic calibration --->"
mkdir -p /mnt/$UTC_DATETIME/data/
cd /mnt/$UTC_DATETIME/data/
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

echo -e "\n\n<--- Running Camera Intrinsic Calibration --->"
mkdir -p /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/
cd /aicv-sensor-drivers/src/sensor_calibrator
python3 -O calibrate-intrinsics.py /mnt/$UTC_DATETIME/data/intrinsics_$UTC_DATETIME/intrinsics_"$UTC_DATETIME"_0.db3 \
                                $RGB_IMAGE_TOPIC \
                                config/staples_april_6x6.yaml \
                                /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/intrinsics_"$UTC_DATETIME".yaml \
                                --out_res_width $PERCY_CAMERA_WIDTH \
                                --out_res_height $PERCY_CAMERA_HEIGHT \
                                --model equidistant --blur-sigma 3 \
                                --inlier-thresh 100 --refinement-method contour --refinement-window 15
if [ $? == 1 ]; then
    echo -e "\n\n### Intrinsic Calibration Failed ###"
    exit 1
else
    echo -e "\n\n### Intrinsic Calibration Successful ###"
fi
mv /tmp/*.png /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/

# # extrinsic calibration
# echo -e "\n\n<--- Collect data for Camera-IMU Extrinsic calibration --->"
# read -rep $'Press L for in-lab calibration (with AprilGrid) \nPress F for in-field calibration (without AprilGrid): ' INPUT

# cd /mnt/$UTC_DATETIME/data/
# echo "Data collection duration: 90 seconds"
# read -p "Press Enter to continue?" ENTER
# echo "Data collection will begin in 5 seconds.."
# sleep 5
# ros2 bag record -d 90 -o extrinsics_$UTC_DATETIME $RGB_IMAGE_TOPIC $IMU_TOPIC &
# EXTRINSICPID=$!
# sleep 91 && kill $EXTRINSICPID && sleep 5

# echo -e "\n\n<--- Running Camera-IMU Extrinsic Calibration --->"
# mkdir -p /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/
# cd /aicv-sensor-drivers/src/sensor_calibrator
# if [[ $INPUT == "L" || $INPUT == "l" ]]; then
#     # in-lab calibration with AprilGrid
#     python3 calibrate-imu-camera.py /mnt/$UTC_DATETIME/data/extrinsics_$UTC_DATETIME/extrinsics_"$UTC_DATETIME"_0.db3 \
#                                     $RGB_IMAGE_TOPIC $IMU_TOPIC \
#                                     --grid-file config/staples_april_6x6.yaml \
#                                     /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/intrinsics_"$UTC_DATETIME".yaml \
#                                     /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/ \
#                                     --blur-sigma 3
# else
#     # in-field calibration using scene features
#     python3 calibrate-imu-camera.py /mnt/$UTC_DATETIME/data/extrinsics_$UTC_DATETIME/extrinsics_"$UTC_DATETIME"_0.db3 \
#                                     $RGB_IMAGE_TOPIC $IMU_TOPIC \
#                                     /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/intrinsics_"$UTC_DATETIME".yaml \
#                                     /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/ \
#                                     --imu-time-correction jitter \
#                                     --cam-time-correction jitter \
#                                     --downsample 1 \
#                                     --debug
# fi

# if [ $? == 1 ]; then
#     echo -e "\n\n### Extrinsic Calibration Failed ###"
#     exit 1
# else
#     echo -e "\n\n### Extrinsic Calibration Successful ###"
    
#     # rename extrinsics
#     mv /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/extrinsics.yaml \
#        /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/extrinsics_"$UTC_DATETIME".yaml
#     mv /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/extrinsics_hd.yaml \
#        /mnt/$UTC_DATETIME/results/extrinsics_$UTC_DATETIME/extrinsics_hd_"$UTC_DATETIME".yaml
# fi

# update mounted directory with latest calibration results
read -rep $'Overwrite previous calibration results in /mnt/latest/ ?: Y/N ' INPUT
if [[ $INPUT == "Y" || $INPUT == "y" ]]; then
    mkdir -p /mnt/latest/

    # # copy imu config
    # cp /aicv-sensor-drivers/src/sensor_calibrator/config/calib_imu.yaml /mnt/latest/imu.yaml

    # copy 4K config
    cp /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/intrinsics_"$UTC_DATETIME".yaml /mnt/latest/camera_4k.yaml
    cp /aicv-sensor-drivers/src/sensor_calibrator/config/vapix_config_day.json /mnt/latest/vapix_config_day.json
    cp /aicv-sensor-drivers/src/sensor_calibrator/config/vapix_config_night.json /mnt/latest/vapix_config_night.json

    # copy HD config
    #cp /mnt/$UTC_DATETIME/results/intrinsics_$UTC_DATETIME/intrinsics_hd_"$UTC_DATETIME".yaml /mnt/latest/camera_hd.yaml
    #cp /aicv-sensor-drivers/src/sensor_calibrator/config/vapix_config_hd.json /mnt/latest/vapix_config_hd.json
fi

echo -e "\n\n##### Calibration complete! Your results are in /mnt/$UTC_DATETIME/results/ #####"