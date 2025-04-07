import pandas as pd
import json
import time
import io

class DataManager:
    def __init__(self):
        """Initialize the DataManager class"""
        pass
    
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
        export_data = {
            "video_info": video_info,
            "processing_info": {
                "timestamp": time.time(),
                "frames_processed": len(frames),
                "total_detections": sum(len(d) for d in detections) if detections else 0
            },
            "frames": []
        }
        
        # Process each frame
        for i, frame in enumerate(frames):
            frame_detections = detections[i] if i < len(detections) else []
            frame_tags = manual_tags.get(i, [])
            
            frame_data = {
                "frame_number": i + 1,
                "timestamp": (i * video_info["frame_interval"]) / video_info["fps"] if video_info["fps"] > 0 else 0,
                "automatic_detections": frame_detections,
                "manual_tags": frame_tags
            }
            
            export_data["frames"].append(frame_data)
        
        return export_data
    
    def export_to_csv(self, export_data):
        """
        Export data to CSV format
        
        Args:
            export_data (dict): Structured data to export
            
        Returns:
            str: CSV data as string
        """
        # Prepare data for CSV format - flatten the structure
        rows = []
        
        video_info = export_data["video_info"]
        
        for frame_data in export_data["frames"]:
            frame_number = frame_data["frame_number"]
            timestamp = frame_data["timestamp"]
            
            # Add automatic detections
            for detection in frame_data["automatic_detections"]:
                rows.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "type": "automatic",
                    "object_class": detection["class"],
                    "confidence": detection["confidence"],
                    "bbox_x": detection["bbox"][0],
                    "bbox_y": detection["bbox"][1],
                    "bbox_width": detection["bbox"][2],
                    "bbox_height": detection["bbox"][3],
                    "description": "",
                })
            
            # Add manual tags
            for tag in frame_data["manual_tags"]:
                rows.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "type": "manual",
                    "object_class": tag["name"],
                    "confidence": 1.0,  # Manual tags have full confidence
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_width": "",
                    "bbox_height": "",
                    "description": tag["description"],
                })
        
        # Create DataFrame and convert to CSV
        if rows:
            df = pd.DataFrame(rows)
            
            # Add video info as metadata rows at the top
            metadata = pd.DataFrame([
                {"frame_number": "VIDEO_INFO", "object_class": "filename", "description": video_info["filename"]},
                {"frame_number": "VIDEO_INFO", "object_class": "resolution", "description": video_info["resolution"]},
                {"frame_number": "VIDEO_INFO", "object_class": "duration", "description": f"{video_info['duration_seconds']:.2f} seconds"},
                {"frame_number": "VIDEO_INFO", "object_class": "total_frames", "description": str(video_info["total_frames"])},
                {"frame_number": "VIDEO_INFO", "object_class": "fps", "description": str(video_info["fps"])},
            ])
            
            # Create a buffer and write the DataFrame to it
            csv_buffer = io.StringIO()
            pd.concat([metadata, df]).to_csv(csv_buffer, index=False)
            
            return csv_buffer.getvalue()
        
        # Return empty CSV if no rows
        return "frame_number,timestamp,type,object_class,confidence,bbox_x,bbox_y,bbox_width,bbox_height,description\n"
    
    def export_to_json(self, export_data):
        """
        Export data to JSON format
        
        Args:
            export_data (dict): Structured data to export
            
        Returns:
            str: JSON data as string
        """
        # Convert to JSON string
        return json.dumps(export_data, indent=2)
    
    def save_to_file(self, data, file_path):
        """
        Save data to a file
        
        Args:
            data (str): Data to save
            file_path (str): Path where to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"Error saving to file: {str(e)}")
            return False
