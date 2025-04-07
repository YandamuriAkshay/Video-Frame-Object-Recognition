import os
import json
import time
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import datetime
import numpy as np
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine with connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Check connection before using from pool
    pool_recycle=3600,   # Recycle connections after 1 hour
    connect_args={
        'connect_timeout': 10,  # Connection timeout in seconds
    }
)

# Configure session
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Function to test database connection with retry
def test_database_connection(max_retries=3, retry_delay=2):
    """Test database connection with retry mechanism"""
    retries = 0
    while retries < max_retries:
        try:
            # Try to connect and run a simple query
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection successful")
            return True
        except OperationalError as e:
            retries += 1
            if retries < max_retries:
                logger.warning(f"Database connection failed. Retrying in {retry_delay}s: {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Database connection failed after {max_retries} attempts: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {str(e)}")
            return False

# Test connection at module load time
test_database_connection()

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
    
    # Define relationships
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
    
    def __repr__(self):
        return f"<Frame(id={self.id}, video_id={self.video_id}, frame_number={self.frame_number})>"
    
    def get_frame_image(self):
        """Convert stored binary data back to a frame image"""
        if self.frame_data:
            nparr = np.frombuffer(self.frame_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None


class Detection(Base):
    """Automatic object detection table"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frames.id'), nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float)
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_width = Column(Integer)
    bbox_height = Column(Integer)
    class_id = Column(Integer)
    
    # Define relationships
    frame = relationship("Frame", back_populates="detections")
    
    def __repr__(self):
        return f"<Detection(id={self.id}, frame_id={self.frame_id}, class_name='{self.class_name}')>"


class ManualTag(Base):
    """Manual tagging table"""
    __tablename__ = 'manual_tags'
    
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frames.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    timestamp = Column(Float)  # When the tag was created
    
    # Define relationships
    frame = relationship("Frame", back_populates="manual_tags")
    
    def __repr__(self):
        return f"<ManualTag(id={self.id}, frame_id={self.frame_id}, name='{self.name}')>"


class DatabaseManager:
    """Database management class for video recognition application"""
    
    def __init__(self, max_init_retries=3):
        """
        Initialize the database and create tables if they don't exist
        
        Args:
            max_init_retries (int): Maximum number of retries for database initialization
        """
        self.engine = engine
        self.Session = Session
        
        # Attempt to initialize database with retry
        retries = 0
        while retries < max_init_retries:
            try:
                # Test connection first
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                
                # If connection successful, create tables
                Base.metadata.create_all(engine)
                logger.info("Database initialized successfully")
                break
            except OperationalError as e:
                retries += 1
                if retries < max_init_retries:
                    logger.warning(f"Database initialization failed. Retrying... ({retries}/{max_init_retries}): {str(e)}")
                    time.sleep(2)
                else:
                    logger.error(f"Database initialization failed after {max_init_retries} attempts: {str(e)}")
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error during initialization: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during database initialization: {str(e)}")
                break
    
    def create_session(self, max_retries=3):
        """
        Create and return a new database session with retry mechanism
        
        Args:
            max_retries (int): Maximum number of retries for session creation
            
        Returns:
            Session: SQLAlchemy session object
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # Try to create a session and test it
                session = self.Session()
                # Test the session with a simple query
                session.execute("SELECT 1").fetchall()
                return session
            except OperationalError as e:
                last_error = e
                retries += 1
                if retries < max_retries:
                    logger.warning(f"Database session creation failed. Retrying ({retries}/{max_retries}): {str(e)}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to create database session after {max_retries} attempts: {str(e)}")
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error creating database session: {str(e)}")
                break
        
        # If we got here, all retries failed
        # Return a session anyway, and let the caller handle any errors
        logger.warning("Returning a session despite connection issues")
        return self.Session()
    
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
        session = self.create_session()
        try:
            # Create new video record
            video = Video(
                filename=video_info.get('filename', 'unknown'),
                resolution=video_info.get('resolution', ''),
                total_frames=video_info.get('total_frames', 0),
                fps=video_info.get('fps', 0),
                duration_seconds=video_info.get('duration_seconds', 0),
                frame_interval=video_info.get('frame_interval', 0),
                max_frames=len(frames),
                processed_timestamp=datetime.datetime.fromtimestamp(
                    video_info.get('processed_timestamp', datetime.datetime.now().timestamp())
                )
            )
            session.add(video)
            session.flush()  # Flush to get the video ID
            
            # Save each frame
            for i, frame_data in enumerate(frames):
                # Encode frame as JPEG for storage
                success, encoded_frame = cv2.imencode('.jpg', frame_data)
                if not success:
                    continue
                
                frame = Frame(
                    video_id=video.id,
                    frame_number=i+1,
                    timestamp=(i * video_info.get('frame_interval', 1)) / video_info.get('fps', 1) 
                              if video_info.get('fps', 0) > 0 else 0,
                    frame_data=encoded_frame.tobytes()
                )
                session.add(frame)
                session.flush()  # Flush to get the frame ID
                
                # Add detections if available
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
                            class_id=det.get('class_id', -1)
                        )
                        session.add(detection)
                
                # Add manual tags if available
                if manual_tags and i in manual_tags:
                    for tag in manual_tags[i]:
                        manual_tag = ManualTag(
                            frame_id=frame.id,
                            name=tag.get('name', ''),
                            description=tag.get('description', ''),
                            timestamp=tag.get('timestamp', datetime.datetime.now().timestamp())
                        )
                        session.add(manual_tag)
            
            # Commit the transaction
            session.commit()
            return video.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_videos(self):
        """
        Get all videos from the database
        
        Returns:
            list: List of video records
        """
        session = self.create_session()
        try:
            return session.query(Video).all()
        finally:
            session.close()
    
    def get_video_by_id(self, video_id):
        """
        Get a video by its ID
        
        Args:
            video_id (int): Video ID
            
        Returns:
            Video: Video record
        """
        session = self.create_session()
        try:
            return session.query(Video).filter(Video.id == video_id).first()
        finally:
            session.close()
    
    def get_frames_by_video_id(self, video_id):
        """
        Get all frames for a specific video
        
        Args:
            video_id (int): Video ID
            
        Returns:
            list: List of Frame records
        """
        session = self.create_session()
        try:
            return session.query(Frame).filter(Frame.video_id == video_id).order_by(Frame.frame_number).all()
        finally:
            session.close()
    
    def get_detections_by_frame_id(self, frame_id):
        """
        Get all detections for a specific frame
        
        Args:
            frame_id (int): Frame ID
            
        Returns:
            list: List of Detection records
        """
        session = self.create_session()
        try:
            return session.query(Detection).filter(Detection.frame_id == frame_id).all()
        finally:
            session.close()
    
    def get_manual_tags_by_frame_id(self, frame_id):
        """
        Get all manual tags for a specific frame
        
        Args:
            frame_id (int): Frame ID
            
        Returns:
            list: List of ManualTag records
        """
        session = self.create_session()
        try:
            return session.query(ManualTag).filter(ManualTag.frame_id == frame_id).all()
        finally:
            session.close()
    
    def add_manual_tag(self, frame_id, name, description=""):
        """
        Add a manual tag to a frame
        
        Args:
            frame_id (int): Frame ID
            name (str): Tag name
            description (str, optional): Tag description
            
        Returns:
            int: ID of the new manual tag
        """
        session = self.create_session()
        try:
            tag = ManualTag(
                frame_id=frame_id,
                name=name,
                description=description,
                timestamp=datetime.datetime.now().timestamp()
            )
            session.add(tag)
            session.commit()
            return tag.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def delete_manual_tag(self, tag_id):
        """
        Delete a manual tag
        
        Args:
            tag_id (int): ID of the tag to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        session = self.create_session()
        try:
            tag = session.query(ManualTag).filter(ManualTag.id == tag_id).first()
            if tag:
                session.delete(tag)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()
    
    def convert_frames_to_numpy(self, db_frames):
        """
        Convert database frames to numpy arrays for processing
        
        Args:
            db_frames (list): List of Frame objects from database
            
        Returns:
            list: List of numpy arrays (frames)
        """
        return [frame.get_frame_image() for frame in db_frames if frame.frame_data]
    
    def convert_detections_to_dict(self, db_detections):
        """
        Convert database detections to dictionary format
        
        Args:
            db_detections (list): List of Detection objects
            
        Returns:
            list: List of detection dictionaries
        """
        return [
            {
                'class': det.class_name,
                'confidence': det.confidence,
                'bbox': [det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height],
                'class_id': det.class_id
            }
            for det in db_detections
        ]
    
    def convert_manual_tags_to_dict(self, db_tags):
        """
        Convert database manual tags to dictionary format
        
        Args:
            db_tags (list): List of ManualTag objects
            
        Returns:
            list: List of tag dictionaries
        """
        return [
            {
                'name': tag.name,
                'description': tag.description,
                'timestamp': tag.timestamp
            }
            for tag in db_tags
        ]
    
    def load_video_data(self, video_id):
        """
        Load complete video data from the database
        
        Args:
            video_id (int): Video ID
            
        Returns:
            tuple: (frames, video_info, detections, manual_tags)
        """
        # Get video record
        video = self.get_video_by_id(video_id)
        if not video:
            return None, None, None, None
        
        # Get all frames for this video
        db_frames = self.get_frames_by_video_id(video_id)
        
        # Convert frames to numpy arrays
        frames = self.convert_frames_to_numpy(db_frames)
        
        # Create video info dictionary
        video_info = {
            'filename': video.filename,
            'resolution': video.resolution,
            'total_frames': video.total_frames,
            'fps': video.fps,
            'duration_seconds': video.duration_seconds,
            'frame_interval': video.frame_interval,
            'processed_timestamp': video.processed_timestamp.timestamp() if video.processed_timestamp else 0
        }
        
        # Get all detections and tags for each frame
        all_detections = []
        all_manual_tags = {}
        
        for i, db_frame in enumerate(db_frames):
            # Get detections
            db_detections = self.get_detections_by_frame_id(db_frame.id)
            all_detections.append(self.convert_detections_to_dict(db_detections))
            
            # Get manual tags
            db_tags = self.get_manual_tags_by_frame_id(db_frame.id)
            all_manual_tags[i] = self.convert_manual_tags_to_dict(db_tags)
        
        return frames, video_info, all_detections, all_manual_tags
    
    def delete_video(self, video_id):
        """
        Delete a video and all its associated data
        
        Args:
            video_id (int): Video ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        session = self.create_session()
        try:
            video = session.query(Video).filter(Video.id == video_id).first()
            if video:
                session.delete(video)  # Cascade will delete frames, detections, and tags
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()