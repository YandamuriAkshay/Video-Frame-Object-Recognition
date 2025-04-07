# Video Frame Object Recognition

A Python-based system for extracting video frames, detecting objects using AI, and enabling manual tagging for visual auditing processes.

## Features

- Upload and process video files
- Extract frames at specified intervals
- Detect objects in frames using YOLOv4-tiny model
- Manually tag objects with custom descriptions
- Store processed videos in a PostgreSQL database
- Export results in CSV or JSON format

## Requirements

- Python 3.8+
- PostgreSQL database
- OpenCV
- Streamlit
- SQLAlchemy
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies 
4. Set up PostgreSQL database

## Running the Application

1. Make sure your PostgreSQL database is running
2. Start the Streamlit server: streamlit run app.py
3. Open your browser and navigate to: http://localhost:8501

## Project Structure

- app.py: Main Streamlit application
- video_processor.py: Handles video frame extraction
- object_detector.py: Performs object detection using YOLOv4-tiny
- data_manager.py: Manages data export functionality
- database.py: Handles database operations
