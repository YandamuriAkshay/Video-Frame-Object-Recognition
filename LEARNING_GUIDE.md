# Video Frame Object Recognition: A Beginner's Guide

This guide provides a comprehensive explanation of the Video Frame Object Recognition project, designed to be accessible for beginners with no prior knowledge of computer vision or web application development.

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Key Functions Explained](#key-functions-explained)
6. [Visual Guide](#visual-guide)
7. [How to Use the Application](#how-to-use-the-application)
8. [Technical Concepts for Beginners](#technical-concepts-for-beginners)

## Project Overview

This project creates a web application that lets users:

1. **Upload video clips** - Share videos from their computer
2. **Extract frames** - Pull out individual images from the video at specific intervals
3. **Detect objects automatically** - Use AI to identify common objects in each frame
4. **Add manual tags** - Manually label things the AI might miss or mislabel
5. **Save and retrieve data** - Store processed videos in a database
6. **Export results** - Download the analysis as CSV or JSON files

This tool simulates systems used in industries like warehousing, retail, and security where visual monitoring and analysis are important.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────►│  Core Processors│────►│  Data Storage   │
│  (Streamlit)    │     │  (Python)       │     │  (PostgreSQL)   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       │                       ▲
        │                       ▼                       │
        │               ┌─────────────────┐             │
        └───────────────│  Data Export    │─────────────┘
                        │  (CSV/JSON)     │
                        └─────────────────┘
```

The system has 4 main parts:

1. **User Interface** - What you see and interact with (built with Streamlit)
2. **Core Processors** - The "brain" that processes videos and detects objects
3. **Data Storage** - Where all the processed videos and tags are saved
4. **Data Export** - How you can download the analysis results

## Core Components

### 1. app.py
The main application file that creates the user interface and connects all components together.

### 2. video_processor.py
Handles extracting frames from uploaded videos.

### 3. object_detector.py
Uses AI (YOLO) to automatically detect objects in images.

### 4. data_manager.py
Manages exporting results to CSV or JSON formats.

### 5. database.py
Handles saving and retrieving data from the PostgreSQL database.

## Data Flow

```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│               │   │               │   │               │
│ Video Upload  │──►│Frame Extraction│──►│Object Detection│
│               │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
                                              │
                                              ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│               │   │               │   │               │
│ Data Export   │◄──│   Database    │◄──│ Manual Tagging │
│               │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

The data moves through the system as follows:

1. **Video Upload** - User selects a video file
2. **Frame Extraction** - System pulls out individual frames at specified intervals
3. **Object Detection** - AI identifies objects in each frame
4. **Manual Tagging** - User adds custom labels to frames
5. **Database Storage** - All information is saved
6. **Data Export** - User downloads results in preferred format

## Key Functions Explained

### Video Processing Functions
- `extract_frames()`: Takes a video file and pulls out individual images at regular intervals
- `save_frame()`: Saves an individual frame as an image file

### Object Detection Functions
- `detect_objects()`: Uses AI to identify common objects in an image
- `_initialize_model()`: Sets up the AI model that recognizes objects

### User Interface Functions
- `reset_session()`: Clears all current data to start fresh
- `navigate_frames()`: Lets users move between different frames of the video

### Database Functions
- `save_video_data()`: Stores all the video data in the database
- `load_video_data()`: Retrieves previously processed videos
- `delete_video()`: Removes a video and all its data from the database

### Data Export Functions
- `prepare_export_data()`: Organizes all the data for exporting
- `export_to_csv()`: Creates a spreadsheet-compatible file
- `export_to_json()`: Creates a structured data file for developers

## Visual Guide

### The Application Workflow

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│                     ┌─────────────────┐                       │
│                     │                 │                       │
│                     │  Start/Reset    │                       │
│                     │                 │                       │
│                     └─────────────────┘                       │
│                              │                                │
│                              ▼                                │
│                     ┌─────────────────┐                       │
│                     │                 │                       │
│                     │  Upload Video   │                       │
│                     │                 │                       │
│                     └─────────────────┘                       │
│                              │                                │
│                              ▼                                │
│                    ┌──────────────────┐                       │
│                    │                  │                       │
│                    │ Configure Settings│                      │
│                    │                  │                       │
│                    └──────────────────┘                       │
│                              │                                │
│                              ▼                                │
│                    ┌──────────────────┐                       │
│                    │                  │                       │
│                    │  Extract Frames  │                       │
│                    │                  │                       │
│                    └──────────────────┘                       │
│                              │                                │
│                              ▼                                │
│                    ┌──────────────────┐                       │
│                    │                  │                       │
│                    │  Detect Objects  │                       │
│                    │                  │                       │
│                    └──────────────────┘                       │
│                              │                                │
│                              ▼                                │
│         ┌────────────────────┴─────────────────┐             │
│         │                                      │             │
│         ▼                                      ▼             │
│ ┌─────────────────┐                   ┌─────────────────┐    │
│ │                 │                   │                 │    │
│ │ Navigate Frames │◄──────────────────►  Add Manual Tags│    │
│ │                 │                   │                 │    │
│ └─────────────────┘                   └─────────────────┘    │
│         │                                      │             │
│         └──────────────────┬───────────────────┘             │
│                            │                                 │
│                            ▼                                 │
│                   ┌──────────────────┐                       │
│                   │                  │                       │
│                   │  Save to Database│                       │
│                   │                  │                       │
│                   └──────────────────┘                       │
│                            │                                 │
│                            ▼                                 │
│                   ┌──────────────────┐                       │
│                   │                  │                       │
│                   │   Export Data    │                       │
│                   │                  │                       │
│                   └──────────────────┘                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## How to Use the Application

### Step 1: Upload a Video
1. Click the "Browse files" button in the sidebar
2. Select a video file (MP4, AVI, MOV, or MKV format)

### Step 2: Configure Frame Extraction
1. Adjust the "Extract every Nth frame" slider (higher = fewer frames)
2. Set the "Maximum frames to extract" value
3. Click "Extract Frames"

### Step 3: Run Object Detection
1. Adjust the "Confidence threshold" slider if needed
2. Click "Detect Objects"

### Step 4: Navigate and Tag Frames
1. Use the navigation buttons to move between frames
2. View automatic detections on the right panel
3. Add manual tags by entering a name and description

### Step 5: Save Your Work
1. Click "Save to Database" to store your processed video

### Step 6: Export Results
1. Select your preferred format (CSV or JSON)
2. Click "Export Tagged Data"
3. Click the download button that appears

## Technical Concepts for Beginners

### What is Object Detection?
Object detection is a computer vision technique that identifies and locates objects within an image. This project uses YOLO (You Only Look Once), a popular algorithm that can recognize common objects like people, cars, animals, and everyday items.

### How Frame Extraction Works
Instead of processing every single frame from a video (which could be thousands for even a short clip), the system extracts frames at regular intervals. For example, if you set the interval to 30, it will take every 30th frame from the video.

### Database Storage
The application uses PostgreSQL, a powerful database system, to store:
- Video metadata (filename, duration, resolution)
- Frame images
- Automatic object detections
- Manual tags

### User Interface Components
The interface is built with Streamlit, a Python library that makes it easy to create web applications. Key components include:
- Sidebar: Contains all controls and settings
- Main panel: Displays the current frame with detection boxes
- Navigation controls: Buttons to move between frames
- Manual tagging section: Form for adding custom tags

### Data Export Formats
- **CSV** (Comma-Separated Values): A simple spreadsheet format that works with Excel
- **JSON** (JavaScript Object Notation): A structured data format often used by developers

## Conclusion

This application demonstrates how computer vision can be used to process video content and extract meaningful information. By combining automatic object detection with manual tagging, it creates a powerful tool for visual data analysis that could be applied in many real-world scenarios.
