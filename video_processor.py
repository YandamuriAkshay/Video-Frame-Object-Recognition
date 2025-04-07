import cv2
import numpy as np
import time
import streamlit as st

class VideoProcessor:
    def __init__(self):
        """Initialize the VideoProcessor class"""
        pass

    def extract_frames(self, video_path, frame_interval=30, max_frames=50):
        """
        Extract frames from a video file at specified intervals
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Extract every Nth frame
            max_frames (int): Maximum number of frames to extract
            
        Returns:
            tuple: (List of extracted frames, Video information dictionary)
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise Exception("Error: Could not open video file.")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Store video info
            video_info = {
                "filename": video_path.split("/")[-1],
                "resolution": f"{width}x{height}",
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration,
                "frame_interval": frame_interval,
                "processed_timestamp": time.time()
            }
            
            # Extract frames
            frames = []
            frame_count = 0
            frame_index = 0
            
            # Create progress indicator
            progress_text = "Extracting frames..."
            progress_bar = st.progress(0)
            
            while frame_count < total_frames and len(frames) < max_frames:
                # Read a frame
                ret, frame = cap.read()
                
                # If frame reading was not successful, break the loop
                if not ret:
                    break
                
                # Extract frame if it's at the specified interval
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    frame_index += 1
                
                # Update progress bar
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                frame_count += 1
            
            # Release resources
            cap.release()
            
            # Log extraction results
            st.info(f"Extracted {len(frames)} frames from {total_frames} total frames in video")
            
            return frames, video_info
            
        except Exception as e:
            st.error(f"Error in frame extraction: {str(e)}")
            # Re-raise the exception for the caller to handle
            raise e

    def save_frame(self, frame, path):
        """
        Save a single frame to disk
        
        Args:
            frame (numpy.ndarray): Frame image data
            path (str): Path where to save the frame
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return cv2.imwrite(path, frame)
        except Exception as e:
            st.error(f"Error saving frame: {str(e)}")
            return False
