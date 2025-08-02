#!/usr/bin/env python3
"""
Create dummy B&W test video for colorization testing
"""

import cv2
import numpy as np
import os

def create_test_bw_video(output_path="test_bw_video.mp4", duration_seconds=2, fps=10):
    """Create a simple B&W test video"""
    
    # Video properties
    width, height = 640, 480
    total_frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating {total_frames} frame B&W test video...")
    
    for frame_num in range(total_frames):
        # Create a simple animated pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(width * 0.3 + (width * 0.4) * (frame_num / total_frames))
        center_y = height // 2
        radius = 50
        
        # Draw circle (grayscale)
        cv2.circle(frame, (center_x, center_y), radius, (128, 128, 128), -1)
        
        # Add some text
        text = f"Frame {frame_num + 1}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Add some geometric shapes
        cv2.rectangle(frame, (50, height-100), (150, height-50), (100, 100, 100), -1)
        
        # Moving line
        line_y = int(height * 0.7 + 50 * np.sin(frame_num * 0.3))
        cv2.line(frame, (0, line_y), (width, line_y), (150, 150, 150), 3)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    print(f"Properties: {width}x{height} @ {fps}fps, {duration_seconds}s duration")
    
    return output_path

def create_reference_image(output_path="test_reference.jpg"):
    """Create a reference color image for Cobra testing"""
    
    width, height = 640, 480
    
    # Create colorful reference image
    ref_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(height):
        for x in range(width):
            ref_image[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient  
                128                        # Constant blue
            ]
    
    # Add some colored shapes
    cv2.circle(ref_image, (width//2, height//2), 80, (0, 255, 255), -1)  # Yellow circle
    cv2.rectangle(ref_image, (50, 50), (150, 150), (255, 0, 0), -1)      # Blue rectangle
    cv2.rectangle(ref_image, (width-150, height-150), (width-50, height-50), (0, 0, 255), -1)  # Red rectangle
    
    cv2.imwrite(output_path, ref_image)
    print(f"Reference image created: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Create test assets
    video_path = create_test_bw_video()
    ref_path = create_reference_image()
    
    print("\nTest assets created successfully!")
    print(f"B&W Video: {video_path}")
    print(f"Reference Image: {ref_path}")
    print("\nYou can now test the colorization features with these files.")
