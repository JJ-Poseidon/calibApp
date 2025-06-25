import cv2
import numpy as np

# pick a small dummy frame
h, w = 64, 64
frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

# upload to GPU
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(frame)

# convert to gray
gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

# download back to CPU
gray = gpu_gray.download()
print("Result shape:", gray.shape)