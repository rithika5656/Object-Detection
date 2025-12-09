"""
MDR Contact Tracing - Object Contamination Detection
Detects when patients touch bottles/cups and marks them contaminated.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Models (auto-download on first run)
yolo_model = YOLO("yolov8l.pt")
pose_model = YOLO("yolov8l-pose.pt")

# Classes: Person=0, Bottle=39, Glass=40, Cup=41
OBJECTS = [39, 40, 41]
PERSONS = [0]
CLASS_NAMES = {0: 'Patient', 39: 'Bottle', 40: 'Glass', 41: 'Cup'}

# Tracking
contaminated = {}

def iou(a, b):
    """Calculate Intersection over Union between two boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def get_palms(keypoints, shape):
    """Calculate palm positions from pose keypoints."""
    palms = []
    h, w = shape[:2]
    if keypoints is None or len(keypoints) < 11: return palms
    
    for elbow_i, wrist_i in [(7, 9), (8, 10)]:  # Left/Right hand
        e, wr = keypoints[elbow_i], keypoints[wrist_i]
        if len(e) > 2 and e[2] > 0.25 and wr[2] > 0.25:
            if wr[0] > 0 and wr[1] > 0:
                px = int(wr[0] + 0.35 * (wr[0] - e[0]))
                py = int(wr[1] + 0.35 * (wr[1] - e[1]))
                palms.append({'center': (px, py), 'box': [max(0,px-55), max(0,py-55), min(w,px+55), min(h,py+55)]})
    return palms

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera"); return

    print("Starting... press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        now = time.time()
        
        # Detect objects and persons
        objs = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=OBJECTS, conf=0.25, verbose=False)[0]
        persons = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=PERSONS, conf=0.5, verbose=False)[0]
        pose = pose_model(frame, conf=0.3, verbose=False)[0]
        
        # Get boxes
        obj_boxes = []
        if objs.boxes is not None and len(objs.boxes) > 0:
            for box, cls, tid in zip(objs.boxes.xyxy.cpu().numpy(), 
                                      objs.boxes.cls.cpu().numpy().astype(int),
                                      (objs.boxes.id.cpu().numpy().astype(int) if objs.boxes.id is not None else range(len(objs.boxes)))):
                obj_boxes.append((tid, cls, box))
        
        patient_boxes = []
        if persons.boxes is not None and len(persons.boxes) > 0:
            for box, cls, tid in zip(persons.boxes.xyxy.cpu().numpy(),
                                      persons.boxes.cls.cpu().numpy().astype(int),
                                      (persons.boxes.id.cpu().numpy().astype(int) if persons.boxes.id is not None else range(len(persons.boxes)))):
                patient_boxes.append((tid, cls, box))
        
        # Get palms
        palms = []
        if pose.keypoints is not None:
            for i in range(len(pose.keypoints)):
                kp = pose.keypoints[i].data[0].cpu().numpy()
                for p in get_palms(kp, frame.shape):
                    # Associate with nearest patient
                    p['patient'] = None
                    for pid, _, pbox in patient_boxes:
                        if pbox[0] <= p['center'][0] <= pbox[2] and pbox[1] <= p['center'][1] <= pbox[3]:
                            p['patient'] = pid; break
                    palms.append(p)
        
        # Collision detection
        for palm in palms:
            for oid, ocls, obox in obj_boxes:
                if iou(palm['box'], obox) > 0.10 and oid not in contaminated:
                    contaminated[oid] = {'patient': palm.get('patient', '?'), 'class': ocls}
                    print(f"[ALERT] Patient {palm.get('patient','?')} touched {CLASS_NAMES.get(ocls,'obj')} (ID:{oid})")
        
        # Draw
        for pid, _, box in patient_boxes:
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(frame, f"Patient {pid}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        for oid, ocls, box in obj_boxes:
            x1,y1,x2,y2 = box.astype(int)
            color = (0,0,255) if oid in contaminated else (0,255,0)
            label = f"{CLASS_NAMES.get(ocls,'obj')} {'CONTAMINATED' if oid in contaminated else 'CLEAN'}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for palm in palms:
            cv2.circle(frame, palm['center'], 10, (0,255,0), -1)
        
        cv2.putText(frame, f"Patients:{len(patient_boxes)} Objects:{len(obj_boxes)} Contaminated:{len(contaminated)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Contact Tracing", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
