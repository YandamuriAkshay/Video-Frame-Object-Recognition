# Video Frame Object Recognition: Technical Documentation

This document provides technical details about the implementation of the Video Frame Object Recognition application, including the purpose of each file, database schema, and key design decisions.

## Project Structure

```
.
├── app.py                 # Main Streamlit application file
├── video_processor.py     # Handles video frame extraction
├── object_detector.py     # Implements object detection using YOLOv4-tiny
├── data_manager.py        # Manages data export functionality
├── database.py            # Handles database operations
├── .streamlit/config.toml # Streamlit configuration
├── requirements.txt       # Project dependencies
├── LEARNING_GUIDE.md      # Beginner-friendly explanation
└── TECHNICAL_DOCUMENTATION.md # This file
```

## Component Details

### 1. app.py

**Purpose**: The main application file that creates the user interface and orchestrates all component interactions.

**Key Features**:
- Streamlit UI with sidebar navigation and controls
- Session state management for temporary data storage
- Navigation between extracted frames
- Integration with all other components
- Database operations (save, load, delete)
- Export functionality

**Main Functions**:
- `reset_session()`: Resets all session variables to initial state
- `navigate_frames()`: Handles frame navigation

### 2. video_processor.py

**Purpose**: Manages video processing and frame extraction.

**Key Features**:
- Video metadata extraction (resolution, fps, duration)
- Frame extraction at specified intervals
- Image processing and saving

**Main Functions**:
- `extract_frames(video_path, frame_interval, max_frames)`: Extracts frames from a video file
- `save_frame(frame, path)`: Saves a single frame to disk

### 3. object_detector.py

**Purpose**: Implements object detection using YOLOv4-tiny model.

**Key Features**:
- YOLO model initialization and loading
- Object detection with configurable confidence threshold
- Bounding box and class information extraction

**Main Functions**:
- `detect_objects(frame, confidence_threshold, nms_threshold)`: Detects objects in a frame
- `_initialize_model()`: Sets up the YOLO model

### 4. data_manager.py

**Purpose**: Manages data export and formatting.

**Key Features**:
- Data preparation for export
- CSV and JSON export formatting

**Main Functions**:
- `prepare_export_data(frames, detections, manual_tags, video_info)`: Prepares data structure for export
- `export_to_csv(export_data)`: Exports data to CSV format
- `export_to_json(export_data)`: Exports data to JSON format
- `save_to_file(data, file_path)`: Saves data to a file

### 5. database.py

**Purpose**: Handles database operations using SQLAlchemy ORM.

**Key Features**:
- Database connection and session management
- Object-relational mapping (ORM) for database tables
- CRUD operations for video data
- Binary storage of frame images

**Main Functions**:
- `save_video_data(frames, video_info, detections, manual_tags)`: Saves complete video data
- `load_video_data(video_id)`: Loads a video and all associated data
- `delete_video(video_id)`: Deletes a video and all associated data
- `get_all_videos()`: Retrieves all videos
- Various getter methods for frames, detections, and tags

## Database Schema

The application uses a PostgreSQL database with the following schema:

### videos
Stores video metadata:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Unique identifier |
| filename | String | Original video filename |
| resolution | String | Video resolution (e.g., "1920x1080") |
| total_frames | Integer | Total number of frames in video |
| fps | Float | Frames per second |
| duration_seconds | Float | Video duration in seconds |
| frame_interval | Integer | Interval used for frame extraction |
| max_frames | Integer | Maximum frames extracted |
| processed_timestamp | DateTime | When the video was processed |

### frames
Stores frame data:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Unique identifier |
| video_id | Integer (FK) | Reference to videos table |
| frame_number | Integer | Frame number in sequence |
| timestamp | Float | Timestamp in video (seconds) |
| frame_data | LargeBinary | Binary representation of the frame image |

### detections
Stores automatic object detections:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Unique identifier |
| frame_id | Integer (FK) | Reference to frames table |
| class_name | String | Detected object class (e.g., "person") |
| confidence | Float | Detection confidence (0-1) |
| bbox_x | Integer | X coordinate of bounding box |
| bbox_y | Integer | Y coordinate of bounding box |
| bbox_width | Integer | Width of bounding box |
| bbox_height | Integer | Height of bounding box |
| class_id | Integer | Numeric class identifier |

### manual_tags
Stores user-added manual tags:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Unique identifier |
| frame_id | Integer (FK) | Reference to frames table |
| name | String | Tag name |
| description | Text | Tag description |
| timestamp | Float | When the tag was created |

## Database Relationships

- **videos** ➝ **frames**: One-to-many (one video has many frames)
- **frames** ➝ **detections**: One-to-many (one frame has many detections)
- **frames** ➝ **manual_tags**: One-to-many (one frame has many manual tags)

## Object Detection Implementation

The project uses YOLOv4-tiny for object detection, a lightweight version of the YOLO (You Only Look Once) algorithm.

### Model Download
The model files are downloaded from public sources:
- Weights: YOLOv4-tiny pre-trained weights
- Configuration: Network architecture definition
- Class names: List of detectable objects (COCO dataset classes)

### Detection Process
1. Image preprocessing (resizing, normalization)
2. Forward pass through neural network
3. Non-maximum suppression to eliminate duplicate detections
4. Confidence filtering to remove low-confidence detections
5. Conversion of results to user-friendly format

## Data Flow

### Video Processing Workflow
1. User uploads video file
2. Application saves it to a temporary location
3. `VideoProcessor.extract_frames()` processes the video
4. Frames are stored in session state
5. Temporary file is deleted

### Object Detection Workflow
1. User clicks "Detect Objects"
2. Each frame is processed by `ObjectDetector.detect_objects()`
3. Detections are stored in session state
4. Results are displayed visually on frames

### Database Workflow
1. User clicks "Save to Database"
2. `DatabaseManager.save_video_data()` stores all information
3. Upon loading, `load_video_data()` retrieves everything
4. Binary frame data is converted back to numpy arrays for display

## Performance Considerations

- Frame extraction is limited to avoid memory issues
- Binary storage of frames in database is optimized for space
- YOLOv4-tiny is used instead of full YOLOv4 for faster processing
- Non-maximum suppression threshold can be adjusted for balance between detection accuracy and performance

## Security Considerations

- User-uploaded files are handled with temporary file objects
- Database connection credentials should be stored as environment variables
- Input validation is used for user-provided frame limits

## Extension Points

The application could be extended with:

1. Support for more object detection models
2. Real-time video processing
3. Custom training for specific object classes
4. Multi-user support
5. Advanced search and filtering of detections and tags

## Deployment Considerations

- Ensure PostgreSQL is installed and configured
- Set database connection environment variables
- Install required Python packages (see requirements.txt)
- Configure Streamlit server settings in .streamlit/config.toml
