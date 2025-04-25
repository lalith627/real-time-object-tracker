import cv2
import numpy as np
import torch
import warnings
from collections import defaultdict
from norfair import Detection, Tracker
import os
import time  # Added for FPS calculation

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize YOLOv5 and get class names
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
CLASS_NAMES = model.names  # Get class names directly from YOLO

# Initialize tracker
tracker = Tracker(
    distance_function="iou",
    distance_threshold=0.3,
    hit_counter_max=15,
    initialization_delay=3
)

# Video capture
cap = cv2.VideoCapture(r'L:\samajh.ai\stock.mov')

# Create resizable window with initial size
cv2.namedWindow('Object Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Tracking', 800, 600)  # Width, Height

# Tracking variables
object_registry = {}  # {id: {'last_box', 'last_pos', 'last_seen', 'moved', 'active', 'class'}}
removed_objects = {}  # {id: {'removal_frame', 'removal_box', 'class'}}
DISAPPEARANCE_FRAMES = 10  # Frames until object considered removed
ALERT_DURATION = 45       # Frames to show removal alert
STATIONARY_THRESHOLD = 5  # Movement threshold in pixels
frame_count = 0
conf_threshold = 0.5      # Confidence threshold
IGNORE_CLASSES = [0]      # Class IDs to ignore for removal (person=0)

# FPS calculation variables
prev_frame_time = 0
new_frame_time = 0

def get_box_center(box):
    """Calculate center point of bounding box"""
    return np.array([(box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2])

def clear_console():
    """Clear console output"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_status(frame_count, active_objects, removed_objects):
    """Print clean status output with limited lines"""
    clear_console()
    print("="*50)
    print(f"FRAME: {frame_count}")
    print("="*50)
    
    # Active objects section
    print("\nACTIVE OBJECTS:")
    print("-"*50)
    if not active_objects:
        print("No active objects detected")
    else:
        for obj_id, info in active_objects.items():
            class_name = CLASS_NAMES.get(info['class'], f"unknown_{info['class']}")
            print(f"{class_name:12s} ID:{obj_id:3d} | Position: {info['position']}")
    
    # Removed objects section
    print("\nRECENTLY REMOVED OBJECTS:")
    print("-"*50)
    recent_removals = [obj for obj in removed_objects.values() 
                      if frame_count - obj['removal_frame'] <= ALERT_DURATION]
    
    if not recent_removals:
        print("No recent removals")
    else:
        for obj in recent_removals:
            class_name = CLASS_NAMES.get(obj['class'], f"unknown_{obj['class']}")
            frames_ago = frame_count - obj['removal_frame']
            print(f"{class_name:12s} | {frames_ago:2d} frames ago")

    # Add some empty lines to prevent scrolling
    print("\n"*3)

def visualize_frame(frame, tracked_objects, removed_objects, fps):
    """Handle all visualization in one function"""
    display_frame = frame.copy()
    
    # Draw FPS counter
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw active objects
    for obj in tracked_objects:
        obj_id = obj.id
        box = obj.estimate
        x1, y1 = box[0].astype(int)
        x2, y2 = box[1].astype(int)
        class_id = object_registry.get(obj_id, {}).get('class', -1)
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        
        # People in blue, other objects in green
        color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Create compact label
        label = f"{class_name[:4]}:{obj_id}"
        cv2.putText(display_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw removal markers
    for obj_id, info in removed_objects.items():
        if info['class'] not in IGNORE_CLASSES and (frame_count - info['removal_frame']) <= ALERT_DURATION:
            removal_box = info['removal_box']
            x1, y1 = removal_box[0].astype(int)
            x2, y2 = removal_box[1].astype(int)
            
            # Red bounding box with transparency
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # "REMOVED" label
            text = f"REMOVED {CLASS_NAMES.get(info['class'], 'Object')}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x1 + (x2 - x1) // 2 - text_width // 2
            text_y = y1 - 15 if y1 > 30 else y2 + 20
            cv2.putText(display_frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return display_frame

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_objects = set()
        active_objects = {}

        # Start FPS calculation
        new_frame_time = time.time()
        
        # YOLO Detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # Prepare detections for Norfair
        norfair_detections = []
        class_info = {}
        for det in detections[detections[:, 4] >= conf_threshold]:
            x1, y1, x2, y2, conf, cls = det
            points = np.array([[x1, y1], [x2, y2]])
            detection = Detection(points=points)
            norfair_detections.append(detection)
            class_info[id(detection)] = int(cls)  # Store class with detection

        # Update tracker
        tracked_objects = tracker.update(detections=norfair_detections)

        # Process tracked objects
        for obj in tracked_objects:
            obj_id = obj.id
            current_objects.add(obj_id)
            box = obj.estimate
            center = get_box_center(box)
            
            # Get class from original detection
            class_id = class_info.get(id(obj.last_detection), -1) if hasattr(obj, 'last_detection') else -1
            
            if obj_id not in object_registry:
                object_registry[obj_id] = {
                    'last_box': box,
                    'last_pos': center,
                    'last_seen': frame_count,
                    'moved': False,
                    'active': True,
                    'class': class_id
                }
            else:
                # Check movement
                movement = np.linalg.norm(center - object_registry[obj_id]['last_pos'])
                object_registry[obj_id]['moved'] = movement > STATIONARY_THRESHOLD
                object_registry[obj_id]['last_box'] = box
                object_registry[obj_id]['last_pos'] = center
                object_registry[obj_id]['last_seen'] = frame_count
                object_registry[obj_id]['class'] = class_id
                
                # If object reappeared after removal
                if obj_id in removed_objects:
                    del removed_objects[obj_id]
            
            # Store active object info
            active_objects[obj_id] = {
                'class': class_id,
                'position': f"[{center[0]:.0f},{center[1]:.0f}]"
            }

        # Check for removed objects (stationary, non-person)
        for obj_id in list(object_registry.keys()):
            if (obj_id not in current_objects and 
                not object_registry[obj_id]['moved'] and
                object_registry[obj_id]['class'] not in IGNORE_CLASSES):
                
                frames_missing = frame_count - object_registry[obj_id]['last_seen']
                if frames_missing >= DISAPPEARANCE_FRAMES and object_registry[obj_id]['active']:
                    object_registry[obj_id]['active'] = False
                    removed_objects[obj_id] = {
                        'removal_frame': frame_count,
                        'removal_box': object_registry[obj_id]['last_box'],
                        'class': object_registry[obj_id]['class']
                    }

        # Cleanup old removal alerts
        removed_objects = {k:v for k,v in removed_objects.items() 
                         if frame_count - v['removal_frame'] <= ALERT_DURATION}

        # Print status to console
        print_status(frame_count, active_objects, removed_objects)

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Visualization
        display_frame = visualize_frame(frame, tracked_objects, removed_objects, fps)
        
        # Display the frame in the resizable window
        cv2.imshow('Object Tracking', display_frame)
        
        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or Q
            break

finally:
    cap.release()
    cv2.destroyAllWindows()