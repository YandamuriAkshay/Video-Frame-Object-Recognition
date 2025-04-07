import cv2
import numpy as np
import streamlit as st
import os
import urllib.request
import tempfile

class ObjectDetector:
    def __init__(self):
        """Initialize the ObjectDetector class with a pre-trained YOLO model"""
        self.model = None
        self.classes = None
        self.initialized = False
        
    def _initialize_model(self):
        """Initialize the YOLO model by downloading necessary files"""
        try:
            # Create a temporary directory for the model files
            self.temp_dir = tempfile.TemporaryDirectory()
            
            # URLs for model files
            model_files = {
                'config': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                'weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
                'classes': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
            }
            
            # Download the files
            file_paths = {}
            for file_type, url in model_files.items():
                local_path = os.path.join(self.temp_dir.name, os.path.basename(url))
                st.info(f"Downloading {file_type} file...")
                urllib.request.urlretrieve(url, local_path)
                file_paths[file_type] = local_path
            
            # Load classes
            with open(file_paths['classes'], 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Load YOLOv4-tiny model
            self.model = cv2.dnn.readNetFromDarknet(file_paths['config'], file_paths['weights'])
            
            # Set backend and target (CPU in this case)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error initializing object detection model: {str(e)}")
            return False
    
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
        # Initialize the model if not already initialized
        if not self.initialized:
            if not self._initialize_model():
                return []
        
        try:
            height, width = frame.shape[:2]
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set the input to the model
            self.model.setInput(blob)
            
            # Get output layer names
            output_layers = self.model.getUnconnectedOutLayersNames()
            
            # Forward pass through the network
            layer_outputs = self.model.forward(output_layers)
            
            # Lists to store detection results
            boxes = []
            confidences = []
            class_ids = []
            
            # Process each output layer
            for output in layer_outputs:
                for detection in output:
                    # Skip the first 5 elements (center_x, center_y, width, height, objectness)
                    scores = detection[5:]
                    # Get the class ID with maximum score
                    class_id = np.argmax(scores)
                    # Get the confidence
                    confidence = scores[class_id]
                    
                    # Filter weak predictions
                    if confidence > confidence_threshold:
                        # YOLO returns coordinates normalized to [0,1]
                        center_x, center_y, box_width, box_height = detection[0:4] * np.array([width, height, width, height])
                        
                        # Calculate top-left corner of the bounding box
                        x = int(center_x - (box_width / 2))
                        y = int(center_y - (box_height / 2))
                        
                        boxes.append([x, y, int(box_width), int(box_height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # Prepare results
            detections = []
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    detection = {
                        'bbox': boxes[i],
                        'confidence': confidences[i],
                        'class_id': class_ids[i],
                        'class': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else f"Unknown-{class_ids[i]}"
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            st.error(f"Error in object detection: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup temporary directory when the object is deleted"""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
