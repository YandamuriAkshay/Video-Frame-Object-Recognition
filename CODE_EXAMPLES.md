# Video Frame Object Recognition: Code Examples Guide

This document provides practical code examples from the project to help beginners understand how each component works.

## Table of Contents
1. [Video Processing Examples](#video-processing-examples)
2. [Object Detection Examples](#object-detection-examples)
3. [Streamlit UI Examples](#streamlit-ui-examples)
4. [Database Examples](#database-examples)
5. [Data Export Examples](#data-export-examples)

## Video Processing Examples

### How Frame Extraction Works

The `extract_frames` function from `video_processor.py` shows how we extract frames at regular intervals:

```python
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
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate how many frames to extract
    frames_to_extract = min(max_frames, total_frames // frame_interval)
    
    # Extract frames at regular intervals
    extracted_frames = []
    for i in range(frames_to_extract):
        # Set the position to the next frame to extract
        frame_position = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read the frame
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    # Create video information dictionary
    video_info = {
        "filename": os.path.basename(video_path),
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration,
        "frame_interval": frame_interval,
        "max_frames": max_frames
    }
    
    return extracted_frames, video_info
```

**Explained for beginners:**
1. We open the video file using OpenCV's `VideoCapture`
2. We get important information about the video (total frames, fps, etc.)
3. We calculate how many frames to extract based on the interval and max frames
4. We loop through the video, jumping ahead by the frame interval each time
5. We read each frame and add it to our list
6. Finally, we return both the frames and information about the video

## Object Detection Examples

### How Object Detection Works

The `detect_objects` function from `object_detector.py` shows how we use YOLO to detect objects:

```python
def detect_objects(self, frame, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect objects in a frame using YOLOv4-tiny
    
    Args:
        frame (numpy.ndarray): The frame to process
        confidence_threshold (float): Minimum confidence for detection
        nms_threshold (float): Non-maximum suppression threshold
        
    Returns:
        list: List of dictionaries with detection results
    """
    # Make sure model is initialized
    self._initialize_model()
    
    # Get image dimensions
    height, width = frame.shape[:2]
    
    # Create a blob from the image (preprocessing)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set the blob as input to the network
    self.net.setInput(blob)
    
    # Run forward pass to get output of the output layers
    layer_names = self.net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    outputs = self.net.forward(output_layers)
    
    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each detection
    for output in outputs:
        for detection in output:
            # Get class scores (starts from 5th element, first 4 are bbox coordinates)
            scores = detection[5:]
            # Get the class with highest score
            class_id = np.argmax(scores)
            # Get the confidence (probability) for the predicted class
            confidence = scores[class_id]
            
            # Filter out weak predictions
            if confidence > confidence_threshold:
                # Scale the bounding box coordinates back relative to the size of the image
                # YOLO returns center (x, y) coordinates of the bounding box
                # followed by the boxes' width and height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Get the top-left corner coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Add to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    # Prepare the results
    detections = []
    
    # Process the final detections
    for i in indices:
        if isinstance(i, list):  # For older versions of OpenCV
            i = i[0]
        
        box = boxes[i]
        x, y, w, h = box
        
        detection = {
            'bbox': [x, y, w, h],
            'confidence': confidences[i],
            'class_id': class_ids[i],
            'class': self.classes[class_ids[i]]
        }
        detections.append(detection)
    
    return detections
```

**Explained for beginners:**
1. We preprocess the image (resize, normalize, etc.)
2. We feed the image through the neural network
3. The network gives us raw detection data
4. We process each detection, filtering out low-confidence ones
5. We apply non-maximum suppression to remove duplicate detections
6. We format the results into a user-friendly dictionary

## Streamlit UI Examples

### Creating the User Interface

Key sections from `app.py` that show how we build the Streamlit interface:

```python
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
    frame_interval = st.slider("Extract every Nth frame", min_value=1, max_value=100, value=30)
    max_frames = st.number_input("Maximum frames to extract", min_value=1, max_value=1000, value=50)
    
    # Process video button
    extract_button = st.button("Extract Frames")
```

**Explained for beginners:**
1. We create a title and description for our app
2. We use the sidebar for controls to keep the main area clean
3. We organize our controls into logical sections
4. We use various widget types (slider, number input, file uploader, button)
5. Each widget returns a value that we can use in our code

### Handling Navigation Between Frames

The `navigate_frames` function shows how we handle navigation:

```python
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
```

**Explained for beginners:**
1. The function takes a direction parameter ('next', 'prev', 'first', or 'last')
2. It checks which direction to move and if the movement is possible
3. It updates the current frame index in the session state
4. Session state is Streamlit's way of preserving values between reruns

## Database Examples

### Database Models

The database models in `database.py` show how we structure our data:

```python
class Video(Base):
    """Video metadata table"""
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    resolution = Column(String(50))
    total_frames = Column(Integer)
    fps = Column(Float)
    duration_seconds = Column(Float)
    frame_interval = Column(Integer)
    max_frames = Column(Integer)
    processed_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Define relationship to Frame
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}')>"

class Frame(Base):
    """Frame data table"""
    __tablename__ = 'frames'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float)
    frame_data = Column(LargeBinary)  # Store frame image as binary
    
    # Define relationships
    video = relationship("Video", back_populates="frames")
    detections = relationship("Detection", back_populates="frame", cascade="all, delete-orphan")
    manual_tags = relationship("ManualTag", back_populates="frame", cascade="all, delete-orphan")
```

**Explained for beginners:**
1. Each class represents a table in the database
2. Columns define what data is stored and its type
3. Relationships define how tables connect to each other
4. This uses SQLAlchemy, which translates Python objects to database records

### Saving Data to the Database

The function that saves video data to the database:

```python
def save_video_data(self, frames, video_info, detections=None, manual_tags=None):
    """
    Save complete video data to the database
    
    Args:
        frames (list): List of video frames (numpy arrays)
        video_info (dict): Dictionary with video metadata
        detections (list, optional): List of detection results for each frame
        manual_tags (dict, optional): Dictionary of manual tags for each frame
        
    Returns:
        int: ID of the saved video record
    """
    try:
        session = self.create_session()
        
        # Create video record
        video = Video(
            filename=video_info.get('filename', 'unknown'),
            resolution=video_info.get('resolution', ''),
            total_frames=video_info.get('total_frames', 0),
            fps=video_info.get('fps', 0),
            duration_seconds=video_info.get('duration_seconds', 0),
            frame_interval=video_info.get('frame_interval', 0),
            max_frames=video_info.get('max_frames', 0)
        )
        session.add(video)
        session.flush()  # Flush to get the ID
        
        # Add frames
        for i, frame_data in enumerate(frames):
            # Convert frame to binary
            _, buffer = cv2.imencode('.png', frame_data)
            frame_binary = buffer.tobytes()
            
            # Calculate timestamp
            timestamp = (i * video_info.get('frame_interval', 0)) / video_info.get('fps', 1) if video_info.get('fps', 0) > 0 else 0
            
            # Create frame record
            frame = Frame(
                video_id=video.id,
                frame_number=i,
                timestamp=timestamp,
                frame_data=frame_binary
            )
            session.add(frame)
            session.flush()  # Flush to get the ID
            
            # Add detections if provided
            if detections and i < len(detections):
                for det in detections[i]:
                    detection = Detection(
                        frame_id=frame.id,
                        class_name=det.get('class', ''),
                        confidence=det.get('confidence', 0),
                        bbox_x=det.get('bbox', [0, 0, 0, 0])[0],
                        bbox_y=det.get('bbox', [0, 0, 0, 0])[1],
                        bbox_width=det.get('bbox', [0, 0, 0, 0])[2],
                        bbox_height=det.get('bbox', [0, 0, 0, 0])[3],
                        class_id=det.get('class_id', 0)
                    )
                    session.add(detection)
            
            # Add manual tags if provided
            if manual_tags and i in manual_tags:
                for tag in manual_tags[i]:
                    manual_tag = ManualTag(
                        frame_id=frame.id,
                        name=tag.get('name', ''),
                        description=tag.get('description', ''),
                        timestamp=tag.get('timestamp', 0)
                    )
                    session.add(manual_tag)
        
        # Commit all changes
        session.commit()
        return video.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
```

**Explained for beginners:**
1. We create a new database session
2. We create a new Video record with metadata
3. For each frame, we:
   - Convert the image to binary format
   - Create a Frame record
   - Add all detections for that frame
   - Add all manual tags for that frame
4. We commit all changes to the database
5. We return the ID of the new video record

## Data Export Examples

### Preparing Data for Export

The function that prepares data for export:

```python
def prepare_export_data(self, frames, detections, manual_tags, video_info):
    """
    Prepare data for export by combining detections and manual tags
    
    Args:
        frames (list): List of video frames
        detections (list): List of detection results for each frame
        manual_tags (dict): Dictionary of manual tags for each frame
        video_info (dict): Dictionary with video information
        
    Returns:
        dict: Structured data ready for export
    """
    # Create export data structure
    export_data = {
        "video_info": video_info,
        "frames": []
    }
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Calculate frame timestamp
        timestamp = (i * video_info['frame_interval']) / video_info['fps'] if video_info['fps'] > 0 else 0
        
        frame_data = {
            "frame_number": i,
            "timestamp": timestamp,
            "automatic_detections": detections[i] if i < len(detections) else [],
            "manual_tags": manual_tags.get(i, [])
        }
        
        export_data["frames"].append(frame_data)
    
    return export_data
```

**Explained for beginners:**
1. We create a dictionary to hold all our export data
2. We add the video information
3. For each frame, we create a frame data dictionary with:
   - Frame number and timestamp
   - All automatic detections for that frame
   - All manual tags for that frame
4. We add each frame's data to our export structure

### Exporting to CSV

The function that exports data to CSV format:

```python
def export_to_csv(self, export_data):
    """
    Export data to CSV format
    
    Args:
        export_data (dict): Structured data to export
        
    Returns:
        str: CSV data as string
    """
    # Create lists to hold all rows
    rows = []
    
    # Create header row
    header = [
        "Frame Number", "Timestamp (s)", "Detection Type", 
        "Object Name", "Confidence", "X", "Y", "Width", "Height",
        "Description"
    ]
    rows.append(header)
    
    # Process each frame
    for frame_data in export_data["frames"]:
        frame_number = frame_data["frame_number"]
        timestamp = frame_data["timestamp"]
        
        # Add automatic detections
        for detection in frame_data["automatic_detections"]:
            x, y, w, h = detection.get("bbox", [0, 0, 0, 0])
            row = [
                frame_number,
                f"{timestamp:.2f}",
                "Automatic",
                detection.get("class", ""),
                f"{detection.get('confidence', 0):.2f}",
                x, y, w, h,
                ""
            ]
            rows.append(row)
        
        # Add manual tags
        for tag in frame_data["manual_tags"]:
            row = [
                frame_number,
                f"{timestamp:.2f}",
                "Manual",
                tag.get("name", ""),
                "",
                "", "", "", "",
                tag.get("description", "")
            ]
            rows.append(row)
    
    # Convert to CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    
    return output.getvalue()
```

**Explained for beginners:**
1. We create a list to hold all rows for our CSV
2. We add a header row with column names
3. For each frame, we add:
   - One row for each automatic detection
   - One row for each manual tag
4. For automatic detections, we include bounding box coordinates
5. For manual tags, we include the description
6. We use the csv module to write all rows to a CSV string

With these code examples, beginners can understand the key concepts and implementation details of the project, which will help them learn and potentially modify or extend the application.
