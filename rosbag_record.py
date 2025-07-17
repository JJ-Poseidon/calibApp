import subprocess
import datetime
import time
import signal

def record_rosbag(topic="/camera1/image_raw", duration=120, save_path="/mnt/"):
    # Generate filename with UTC datetime
    utc_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    bag_name = f"{save_path}{utc_datetime}/results"

    # ros2 bag record command
    cmd = [
        "ros2", "bag", "record",
        "-o", bag_name,
        topic
    ]

    print(f"Starting rosbag record on topic {topic} for {duration} seconds...")
    # Start subprocess
    proc = subprocess.Popen(cmd)

    try:
        # Wait for the specified duration
        time.sleep(duration)
    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    finally:
        # Terminate the recording process gracefully
        print("Stopping rosbag record...")
        proc.send_signal(signal.SIGINT)
        proc.wait()

    print(f"Rosbag saved to {bag_name}")
