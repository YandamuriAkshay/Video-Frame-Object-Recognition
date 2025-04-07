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
st.title("🎬 Video Frame Object Recognition")
st.markdown("Extract frames from videos, detect objects, and manually tag them for visual auditing.")

# Sidebar for controls and settings
with st.sidebar:
    st.header("🎮 Controls")
    
    # File upload section
    st.markdown("<div class='settings-section'><span class='icon-upload'></span><strong>1. Upload Video</strong></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    # Frame extraction settings
    st.markdown("<div class='settings-section'><span class='icon-settings'></span><strong>2. Frame Extraction Settings</strong></div>", unsafe_allow_html=True)
    frame_interval = st.slider("Extract every Nth frame", min_value=1, max_value=100, value=30, 
                               help="Higher values = fewer frames extracted")
    max_frames = st.number_input("Maximum frames to extract", min_value=1, max_value=1000, value=50)
    
    # Process video button
    extract_button = st.button("🔍 Extract Frames")
    
    # Object detection section
    st.markdown("<div class='settings-section'><span class='icon-detection'></span><strong>3. Object Detection</strong></div>", unsafe_allow_html=True)
    confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    detect_button = st.button("🤖 Detect Objects", disabled=not st.session_state.extraction_complete)
    
    # Export section
    st.markdown("<div class='settings-section'><span class='icon-export'></span><strong>4. Export Results</strong></div>", unsafe_allow_html=True)
    export_format = st.selectbox("Export format", ["CSV", "JSON"])
    export_button = st.button("📊 Export Data", disabled=not st.session_state.detection_complete)
    
    # Database section
    st.markdown("<div class='settings-section'><span class='icon-database'></span><strong>5. Database</strong></div>", unsafe_allow_html=True)
    
    # Save to database button
    save_db_button = st.button("💾 Save to Database", 
                              disabled=not (st.session_state.extraction_complete and st.session_state.detection_complete))
    
    # Saved videos selector
    saved_videos = db_manager.get_all_videos()
    video_options = ["Select a saved video..."] + [f"{v.id}: {v.filename} ({v.processed_timestamp.strftime('%Y-%m-%d %H:%M')})" 
                                                for v in saved_videos]
    selected_video = st.selectbox("Load saved video", options=video_options)
    
    col1, col2 = st.columns(2)
    with col1:
        load_video_button = st.button("📂 Load", 
                                    disabled=(selected_video == "Select a saved video..."))
    with col2:
        delete_video_button = st.button("🗑️ Delete", 
                                    disabled=(selected_video == "Select a saved video..."))
    
    # Reset button
    st.markdown("<div class='settings-section'><span class='icon-reset'></span><strong>6. Reset</strong></div>", unsafe_allow_html=True)
    reset_button = st.button("🔄 Reset All")

# Main content area
if reset_button:
    reset_session()
    st.markdown('<div class="success-message">🔄 All data has been reset.</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="success-message">✅ Successfully extracted {len(frames)} frames from video.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">❌ No frames could be extracted from the video.</div>', unsafe_allow_html=True)
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
            st.markdown(f'<div class="success-message">🤖 Object detection completed on {len(all_detections)} frames.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error during object detection: {str(e)}</div>', unsafe_allow_html=True)

# Display frame and detection results
if st.session_state.extraction_complete:
    st.markdown("<h2>🖼️ Frame Viewer</h2>", unsafe_allow_html=True)
    
    # Create columns for the viewer
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Frame container with custom styling
        st.markdown('<div class="content-card frame-container">', unsafe_allow_html=True)
        
        # Navigation controls with icons and styling
        st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                    
                    # Draw rectangle and label with bright colors for visibility on dark theme
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow border
                    # Add background to text for better readability
                    text_size = cv2.getTextSize(f"{label} {confidence:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(current_frame, (x, y - 20), (x + text_size[0], y), (0, 0, 0), -1)
                    cv2.putText(current_frame, f"{label} {confidence:.2f}", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Convert from BGR to RGB for display
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame with custom caption
            frame_caption = f"Frame {current_index + 1} of {len(st.session_state.frames)}"
            if st.session_state.video_info:
                frame_timestamp = f"{(current_index * st.session_state.video_info['frame_interval']) / st.session_state.video_info['fps']:.2f}"
                frame_caption += f" | Time: {frame_timestamp}s"
            
            st.image(current_frame_rgb, caption=frame_caption, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Frame details in a styled card
        if st.session_state.video_info and st.session_state.frames:
            current_index = st.session_state.current_frame_index
            with st.expander("Frame Details"):
                frame_details = {
                    "Frame Number": current_index + 1,
                    "Total Frames": len(st.session_state.frames),
                    "Timestamp": f"{(current_index * st.session_state.video_info['frame_interval']) / st.session_state.video_info['fps']:.2f} seconds",
                    "Resolution": f"{st.session_state.video_info.get('width', 'N/A')}x{st.session_state.video_info.get('height', 'N/A')}"
                }
                
                st.json(frame_details)
    
    with col2:
        # Manual tagging section with improved styling
        st.markdown("<h3>🏷️ Manual Tagging</h3>", unsafe_allow_html=True)
        
        current_index = st.session_state.current_frame_index
        
        # Display automatic detections with styled containers
        if st.session_state.detection_complete:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown("🤖 **Automatic Detections:**")
            current_detections = st.session_state.detections[current_index]
            if current_detections:
                for i, detection in enumerate(current_detections):
                    st.markdown(
                        f'<div class="detection-item">{i+1}. {detection["class"]} '
                        f'<span style="color:#BB86FC">({detection["confidence"]:.2f})</span></div>', 
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="detection-item">No objects detected in this frame.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Manual tag input with styled form
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("✏️ **Add Manual Tag:**")
        tag_name = st.text_input("Object name", key=f"tag_name_{current_index}")
        tag_desc = st.text_area("Description (optional)", key=f"tag_desc_{current_index}", height=80)
        
        if st.button("➕ Add Tag", key=f"add_tag_{current_index}"):
            if tag_name:
                if current_index not in st.session_state.manual_tags:
                    st.session_state.manual_tags[current_index] = []
                
                # Add the tag to the current frame
                st.session_state.manual_tags[current_index].append({
                    "name": tag_name,
                    "description": tag_desc,
                    "timestamp": time.time()
                })
                st.markdown(f'<div class="success-message">✅ Added tag "{tag_name}" to frame {current_index + 1}</div>', unsafe_allow_html=True)
                
                # Clear the input fields by forcing a rerun
                st.rerun()
            else:
                st.markdown('<div class="error-message">❌ Please enter an object name.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display manual tags for this frame in styled containers
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("📋 **Manual Tags:**")
        if current_index in st.session_state.manual_tags and st.session_state.manual_tags[current_index]:
            for i, tag in enumerate(st.session_state.manual_tags[current_index]):
                st.markdown(f'<div class="tag-container">', unsafe_allow_html=True)
                col_tag, col_delete = st.columns([4, 1])
                with col_tag:
                    st.markdown(f"**{tag['name']}**")
                    if tag['description']:
                        st.markdown(f"<em>{tag['description']}</em>", unsafe_allow_html=True)
                with col_delete:
                    if st.button("🗑️", key=f"delete_tag_{current_index}_{i}"):
                        st.session_state.manual_tags[current_index].pop(i)
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pending-step">No manual tags for this frame.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
            
        st.markdown(f'<div class="success-message">📊 Data exported successfully in {export_format} format.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error exporting data: {str(e)}</div>', unsafe_allow_html=True)

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
            
            st.markdown(f'<div class="success-message">💾 Video data saved to database with ID: {video_id}</div>', unsafe_allow_html=True)
            st.session_state.saved_videos = db_manager.get_all_videos()
            st.rerun()
    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error saving to database: {str(e)}</div>', unsafe_allow_html=True)

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
                
                st.markdown(f'<div class="success-message">📂 Successfully loaded video data from database (ID: {video_id})</div>', unsafe_allow_html=True)
                st.rerun()
            else:
                st.markdown('<div class="error-message">❌ Failed to load video data from database</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error loading from database: {str(e)}</div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="success-message">🗑️ Video with ID {video_id} deleted from database</div>', unsafe_allow_html=True)
                # Reset confirmation flag
                st.session_state.confirmed_delete = False
                # Refresh saved videos list
                st.session_state.saved_videos = db_manager.get_all_videos()
                st.rerun()
            else:
                st.markdown(f'<div class="error-message">❌ Failed to delete video with ID {video_id}</div>', unsafe_allow_html=True)
        else:
            # First ask for confirmation with styled warning
            st.markdown(f'<div style="background-color:#3A3A1E; color:#FFC107; padding:1rem; border-radius:8px; border-left:4px solid #FFC107;">⚠️ Are you sure you want to delete video with ID {video_id}? This cannot be undone.</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete"):
                    st.session_state.confirmed_delete = True
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.rerun()
    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error deleting from database: {str(e)}</div>', unsafe_allow_html=True)
