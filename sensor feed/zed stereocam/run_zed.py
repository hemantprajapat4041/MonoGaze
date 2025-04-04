#!/usr/bin/env python3

import pyzed.sl as sl
import numpy as np
import cv2
import os
import time
from datetime import datetime
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./sensor_out/zed_output', help='Base output directory')
    parser.add_argument('--duration', type=int, default=0, help='Duration in seconds (0 for infinite)')
    args = parser.parse_args()
    
    # Create output directories
    image_dir = os.path.join(args.output_dir, 'images')
    depth_dir = os.path.join(args.output_dir, 'depth')
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    print(f"Saving images to: {image_dir}")
    print(f"Saving depth data to: {depth_dir}")
    
    # Initialize ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 for better performance
    init_params.camera_fps = 60  # Set camera FPS to 20
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA for better depth quality
    init_params.coordinate_units = sl.UNIT.METER  # Set units to meter
    init_params.depth_minimum_distance = 0.15
    init_params.depth_maximum_distance = 40.0
    
    # Create ZED camera object
    zed = sl.Camera()
    
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        return
    
    # Create runtime parameters for depth
    runtime_params = sl.RuntimeParameters()
    
    # Create Mat objects for images and depth
    image = sl.Mat()
    depth = sl.Mat()
    
    # Calculate delay between frames to achieve 20 FPS
    frame_delay = 1.0 / 20.0  # 50ms for 20fps
    
    # Set up recording variables
    start_time = time.time()
    frame_count = 0
    
    try:
        print("Recording started. Press Ctrl+C to stop.")
        
        while True:
            # Check if duration specified and exceeded
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break
                
            loop_start_time = time.time()
            
            # Grab a new frame from the ZED
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Get current timestamp for filename
                timestamp = datetime.now().strftime("%H:%M:%S.%f")
                
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img_np = image.get_data()
                
                # Retrieve depth map
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_np = depth.get_data()
                
                # Save image as PNG
                image_path = os.path.join(image_dir, f"{timestamp}.png")
                cv2.imwrite(image_path, img_np)
                
                # Save depth as NPY
                depth_path = os.path.join(depth_dir, f"{timestamp}.npy")
                np.save(depth_path, depth_np)
                
                frame_count += 1
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Captured {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
            
            
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        # Close the camera
        zed.close()
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\nRecording finished.")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {frame_count / elapsed:.2f}")

if __name__ == "__main__":
    main()