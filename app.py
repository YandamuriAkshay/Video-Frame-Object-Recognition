import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time

from video_processor import VideoProcessor
from object_detector import ObjectDetector
from data_manager import DataManager
from database import DatabaseManager

# Set page configuration
st.set_page_config(
    page_title="Video Frame Object Recognition",
    page_icon="🎬",
    layout="wide"
)

# Load custom CSS
def load_css():
    with open(".streamlit/style.css", "r") as f:
        css = f.read()
    return st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Apply custom CSS
load_css()

# Initialize session state variables if they don't exist
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'current_frame_index' not in st.session_state:
    st.session_state.current_frame_index = 0
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'manual_tags' not in st.session_state:
    st.session_state.manual_tags = {}
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'extraction_complete' not in st.session_state:
    st.session_state.extraction_complete = False
if 'detection_complete' not in st.session_state:
    st.session_state.detection_complete = False

# Initialize objects
video_processor = VideoProcessor()
object_detector = ObjectDetector()
data_manager = DataManager()
db_manager = DatabaseManager()

# Initialize session state for database
if 'saved_videos' not in st.session_state:
    st.session_state.saved_videos = []
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None


def reset_session():
    """Reset all session variables to initial state"""
    st.session_state.frames = []
    st.session_state.current_frame_index = 0
    st.session_state.detections = []
    st.session_state.manual_tags = {}
    st.session_state.video_info = None
    st.session_state.extraction_complete = False
    st.session_state.detection_complete = False
    st.session_state.current_video_id = None


def navigate_frames(direction):
    """Navigate between frames"""
    if direction == "next" and st.session_state.current_frame_index < len(st.session_state.frames) - 1:
        st.session_state.current_frame_index += 1
    elif direction == "prev" and st.session_state.current_frame_index > 0:
        st.session_state.current_frame_index -= 1
    elif direction == "first":
        st.session_state.current_frame_index = 0
    elif direction == "last":
        st.session_state.current_frame_index = len(st.session_state.frames) - 1


# Main application layout
st.title("Video Frame Object Recognition")
st.markdown("Extract frames from videos, detect objects, and manually tag them.")

# Sidebar for controls and settings
with st.sidebar:
    st.header("Controls")
    
    # File upload section
    st.subheader("1. Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    # Frame extraction settings
    st.subheader("2. Frame Extraction Settings")
    frame_interval = st.slider("Extract every Nth frame", min_value=1, max_value=100, value=30, 
                               help="Higher values = fewer frames extracted")
    max_frames = st.number_input("Maximum frames to extract", min_value=1, max_value=1000, value=50)
    
    # Process video button
    extract_button = st.button("Extract Frames")
    
    # Object detection section
    st.subheader("3. Object Detection")
    confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    detect_button = st.button("Detect Objects", disabled=not st.session_state.extraction_complete)
    
    # Export section
    st.subheader("4. Export Results")
    export_format = st.selectbox("Export format", ["CSV", "JSON"])
    export_button = st.button("Export Tagged Data", disabled=not st.session_state.detection_complete)
    
    # Database section
    st.subheader("5. Database")
    
    # Save to database button
    save_db_button = st.button("Save to Database", 
                              disabled=not (st.session_state.extraction_complete and st.session_state.detection_complete))
    
    # Saved videos selector
    saved_videos = db_manager.get_all_videos()
    video_options = ["Select a saved video..."] + [f"{v.id}: {v.filename} ({v.processed_timestamp.strftime('%Y-%m-%d %H:%M')})" 
                                                for v in saved_videos]
    selected_video = st.selectbox("Load saved video", options=video_options)
    
    load_video_button = st.button("Load Selected Video", 
                                 disabled=(selected_video == "Select a saved video..."))
    
    delete_video_button = st.button("Delete Selected Video", 
                                   disabled=(selected_video == "Select a saved video..."))
    
    # Reset button
    st.subheader("6. Reset")
    reset_button = st.button("Reset All")

# Main content area
if reset_button:
    reset_session()
    st.success("All data has been reset.")
    st.rerun()

# Handle video upload and frame extraction
if uploaded_file is not None and extract_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    with st.spinner('Extracting frames from video...'):
        try:
            # Extract frames and get video information
            frames, video_info = video_processor.extract_frames(
                video_path, frame_interval, max_frames
            )
            
            if frames:
                st.session_state.frames = frames
                st.session_state.video_info = video_info
                st.session_state.extraction_complete = True
                st.session_state.manual_tags = {i: [] for i in range(len(frames))}
                st.success(f"Successfully extracted {len(frames)} frames from video.")
            else:
                st.error("No frames could be extracted from the video.")
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)

# Handle object detection
if st.session_state.extraction_complete and detect_button:
    with st.spinner('Detecting objects in frames...'):
        try:
            all_detections = []
            progress_bar = st.progress(0)
            
            for i, frame in enumerate(st.session_state.frames):
                # Detect objects in the frame
                detections = object_detector.detect_objects(frame, confidence_threshold)
                all_detections.append(detections)
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(st.session_state.frames))
            
            st.session_state.detections = all_detections
            st.session_state.detection_complete = True
            st.success(f"Object detection completed on {len(all_detections)} frames.")
        except Exception as e:
            st.error(f"Error during object detection: {str(e)}")

# Display frame and detection results
if st.session_state.extraction_complete:
    st.subheader("Frame Viewer")
    
    # Create columns for the viewer
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Navigation controls
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
        with nav_col1:
            st.button("⏮️ First", on_click=navigate_frames, args=("first",), 
                      disabled=st.session_state.current_frame_index == 0)
        with nav_col2:
            st.button("◀️ Previous", on_click=navigate_frames, args=("prev",), 
                      disabled=st.session_state.current_frame_index == 0)
        with nav_col3:
            st.button("Next ▶️", on_click=navigate_frames, args=("next",), 
                      disabled=st.session_state.current_frame_index == len(st.session_state.frames) - 1)
        with nav_col4:
            st.button("Last ⏭️", on_click=navigate_frames, args=("last",), 
                      disabled=st.session_state.current_frame_index == len(st.session_state.frames) - 1)
        
        # Frame display
        if st.session_state.frames:
            current_index = st.session_state.current_frame_index
            
            # Get current frame
            current_frame = st.session_state.frames[current_index].copy()
            
            # Draw detection boxes if available
            if st.session_state.detection_complete:
                current_detections = st.session_state.detections[current_index]
                for detection in current_detections:
                    x, y, w, h = detection['bbox']
                    label = detection['class']
                    confidence = detection['confidence']
                    
                    # Draw rectangle and label
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(current_frame, f"{label} {confidence:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert from BGR to RGB for display
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            st.image(current_frame_rgb, caption=f"Frame {current_index + 1} of {len(st.session_state.frames)}", 
                     use_column_width=True)
            
            # Frame details
            frame_details = {
                "Frame Number": current_index + 1,
                "Timestamp": f"{(current_index * st.session_state.video_info['frame_interval']) / st.session_state.video_info['fps']:.2f} seconds"
            }
            
            # Display frame information
            st.json(frame_details)
    
    with col2:
        # Manual tagging section
        st.subheader("Manual Tagging")
        
        current_index = st.session_state.current_frame_index
        
        # Display automatic detections
        if st.session_state.detection_complete:
            st.markdown("**Automatic Detections:**")
            current_detections = st.session_state.detections[current_index]
            if current_detections:
                for i, detection in enumerate(current_detections):
                    st.text(f"{i+1}. {detection['class']} ({detection['confidence']:.2f})")
            else:
                st.text("No objects detected in this frame.")
        
        # Manual tag input
        st.markdown("**Add Manual Tag:**")
        tag_name = st.text_input("Object name", key=f"tag_name_{current_index}")
        tag_desc = st.text_area("Description (optional)", key=f"tag_desc_{current_index}")
        
        if st.button("Add Tag", key=f"add_tag_{current_index}"):
            if tag_name:
                if current_index not in st.session_state.manual_tags:
                    st.session_state.manual_tags[current_index] = []
                
                # Add the tag to the current frame
                st.session_state.manual_tags[current_index].append({
                    "name": tag_name,
                    "description": tag_desc,
                    "timestamp": time.time()
                })
                st.success(f"Added tag '{tag_name}' to frame {current_index + 1}")
                
                # Clear the input fields by forcing a rerun
                st.rerun()
            else:
                st.error("Please enter an object name.")
        
        # Display manual tags for this frame
        st.markdown("**Manual Tags:**")
        if current_index in st.session_state.manual_tags and st.session_state.manual_tags[current_index]:
            for i, tag in enumerate(st.session_state.manual_tags[current_index]):
                with st.container():
                    col_tag, col_delete = st.columns([4, 1])
                    with col_tag:
                        st.markdown(f"**{tag['name']}**")
                        if tag['description']:
                            st.text(tag['description'])
                    with col_delete:
                        if st.button("🗑️", key=f"delete_tag_{current_index}_{i}"):
                            st.session_state.manual_tags[current_index].pop(i)
                            st.rerun()
        else:
            st.text("No manual tags for this frame.")

# Handle export functionality
if export_button and st.session_state.detection_complete:
    try:
        export_data = data_manager.prepare_export_data(
            st.session_state.frames,
            st.session_state.detections,
            st.session_state.manual_tags,
            st.session_state.video_info
        )
        
        if export_format == "CSV":
            csv_data = data_manager.export_to_csv(export_data)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="video_object_recognition_results.csv",
                mime="text/csv"
            )
        else:  # JSON
            json_data = data_manager.export_to_json(export_data)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="video_object_recognition_results.json",
                mime="application/json"
            )
            
        st.success(f"Data exported successfully in {export_format} format.")
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")

# Display video info if available
if st.session_state.video_info:
    with st.expander("Video Information"):
        st.json(st.session_state.video_info)

# Handle save to database functionality
if save_db_button and st.session_state.detection_complete:
    try:
        with st.spinner('Saving to database...'):
            # Save video data to database
            video_id = db_manager.save_video_data(
                st.session_state.frames,
                st.session_state.video_info,
                st.session_state.detections,
                st.session_state.manual_tags
            )
            
            # Set current video ID
            st.session_state.current_video_id = video_id
            
            st.success(f"Video data saved to database with ID: {video_id}")
            st.session_state.saved_videos = db_manager.get_all_videos()
            st.rerun()
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")

# Handle load from database functionality
if load_video_button and selected_video != "Select a saved video...":
    try:
        with st.spinner('Loading from database...'):
            # Extract video ID from selection
            video_id = int(selected_video.split(":")[0])
            
            # Load video data from database
            frames, video_info, detections, manual_tags = db_manager.load_video_data(video_id)
            
            # Update session state with loaded data
            if frames and video_info:
                st.session_state.frames = frames
                st.session_state.video_info = video_info
                st.session_state.detections = detections
                st.session_state.manual_tags = manual_tags
                st.session_state.current_frame_index = 0
                st.session_state.extraction_complete = True
                st.session_state.detection_complete = True
                st.session_state.current_video_id = video_id
                
                st.success(f"Successfully loaded video data from database (ID: {video_id})")
                st.rerun()
            else:
                st.error("Failed to load video data from database")
    except Exception as e:
        st.error(f"Error loading from database: {str(e)}")

# Handle delete from database functionality
if delete_video_button and selected_video != "Select a saved video...":
    try:
        # Extract video ID from selection
        video_id = int(selected_video.split(":")[0])
        
        # Show confirmation dialog
        if st.session_state.get('confirmed_delete', False):
            # If already confirmed, delete the video
            success = db_manager.delete_video(video_id)
            
            if success:
                st.success(f"Video with ID {video_id} deleted from database")
                # Reset confirmation flag
                st.session_state.confirmed_delete = False
                # Refresh saved videos list
                st.session_state.saved_videos = db_manager.get_all_videos()
                st.rerun()
            else:
                st.error(f"Failed to delete video with ID {video_id}")
        else:
            # First ask for confirmation
            st.warning(f"Are you sure you want to delete video with ID {video_id}? This cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete"):
                    st.session_state.confirmed_delete = True
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.rerun()
    except Exception as e:
        st.error(f"Error deleting from database: {str(e)}")
