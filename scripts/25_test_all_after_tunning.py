from ultralytics import YOLO
import cv2
from collections import deque, Counter
import math

# ==============================
# LOAD MODELS
# ==============================
helmet_model = YOLO("models/final/helmet_best_yolov8s.pt")
mask_model = YOLO("runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt")
gasmask_model = YOLO("runs/detect/models/finetune/gasmask_yolov8s_finetuned_SAFE/weights/best.pt")

# ==============================
# OPEN CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit(1)

print("📷 Starting MULTI-PERSON PPE detection...")
print("Press 'q' to quit")

# ==============================
# MEMORY
# ==============================
person_histories = {}  # each person gets their own history


# ==============================
# HELPERS
# ==============================
def stable_vote(history, default_value):
    if len(history) == 0:
        return default_value
    return Counter(history).most_common(1)[0][0]


def center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def make_person_box_from_face(face_box, frame_w, frame_h):
    x1, y1, x2, y2 = face_box[:4]
    fw = x2 - x1
    fh = y2 - y1

    px1 = max(0, x1 - int(fw * 0.8))
    py1 = max(0, y1 - int(fh * 1.2))
    px2 = min(frame_w, x2 + int(fw * 0.8))
    py2 = min(frame_h, y2 + int(fh * 2.8))

    return (px1, py1, px2, py2)


def get_person_id(person_box):
    cx, cy = center(person_box)
    return f"{cx//50}_{cy//50}"   # simple grid-based ID


# ==============================
# PROCESS FRAME
# ==============================
def process_frame(frame):
    global person_histories

    output = frame.copy()
    h, w, _ = frame.shape

    # Run all models
    helmet_result = helmet_model(frame)[0]
    mask_result = mask_model(frame)[0]
    gasmask_result = gasmask_model(frame)[0]

    helmet_hats = []
    mask_detections = []
    gasmask_boxes = []

    # ==============================
    # HELMET
    # ==============================
    for box in helmet_result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.15:
            continue

        if cls_id == 0:  # hat
            helmet_hats.append((x1, y1, x2, y2, conf))

    # ==============================
    # MASK
    # 0 = with_mask
    # 1 = without_mask
    # 2 = incorrect_mask
    # ==============================
    for box in mask_result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.15:
            continue

        mask_detections.append((x1, y1, x2, y2, cls_id, conf))

# ==============================
# GAS MASK
# ==============================
    for box in gasmask_result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print("GASMASK DETECT:", cls_id, conf)  # debug

        if conf < 0.25:
            continue

        if cls_id == 1:  # gasmask
            gasmask_boxes.append((x1, y1, x2, y2, conf))
    # ==============================
    # BUILD PERSON BOXES
    # We use mask + gasmask + helmet to create people
    # ==============================
    person_candidates = []

    # From mask detections
    for m in mask_detections:
        person_candidates.append(make_person_box_from_face(m, w, h))

    # From gasmask detections
    for g in gasmask_boxes:
        person_candidates.append(make_person_box_from_face(g, w, h))

    # From helmet detections
    for hbox in helmet_hats:
        x1, y1, x2, y2, _ = hbox
        hw = x2 - x1
        hh = y2 - y1

        px1 = max(0, x1 - int(hw * 0.5))
        py1 = max(0, y1 - int(hh * 0.2))
        px2 = min(w, x2 + int(hw * 0.5))
        py2 = min(h, y2 + int(hh * 2.2))

        person_candidates.append((px1, py1, px2, py2))

    # Remove near-duplicate person boxes
    final_persons = []
    for candidate in person_candidates:
        cx1, cy1 = center(candidate)
        too_close = False

        for existing in final_persons:
            cx2, cy2 = center(existing)
            if distance((cx1, cy1), (cx2, cy2)) < 120:
                too_close = True
                break

        if not too_close:
            final_persons.append(candidate)

    # ==============================
    # PER PERSON PPE ANALYSIS
    # ==============================
    safe_count = 0
    unsafe_count = 0

    for person_box in final_persons:
        px1, py1, px2, py2 = person_box
        person_id = get_person_id(person_box)

        if person_id not in person_histories:
            person_histories[person_id] = {
                "helmet": deque(maxlen=10),
                "mask": deque(maxlen=10),
                "gasmask": deque(maxlen=10),
                "safe": deque(maxlen=10)
            }

        hist = person_histories[person_id]

        current_helmet = "No Helmet"
        current_mask = "No Mask"
        current_gasmask = "No Gas Mask"

        # ------------------------------
        # Match helmet to this person
        # ------------------------------
        for (hx1, hy1, hx2, hy2, hconf) in helmet_hats:
            cx, cy = center((hx1, hy1, hx2, hy2))

            if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.55:
                current_helmet = "Helmet"
                break

        # ------------------------------
        # Match mask to this person
        # ------------------------------
        best_mask_conf = 0
        best_mask_label = "No Mask"

        for (mx1, my1, mx2, my2, mcls, mconf) in mask_detections:
            cx, cy = center((mx1, my1, mx2, my2))

            if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.75:
                if mconf > best_mask_conf:
                    best_mask_conf = mconf
                    if mcls == 0:
                        best_mask_label = "With Mask"
                    elif mcls == 1:
                        best_mask_label = "Without Mask"
                    elif mcls == 2:
                        best_mask_label = "Incorrect Mask"

        current_mask = best_mask_label

        # ------------------------------
        # Match gasmask to this person
        # ------------------------------
        for (gx1, gy1, gx2, gy2, gconf) in gasmask_boxes:
            cx, cy = center((gx1, gy1, gx2, gy2))

            if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.75:
                current_gasmask = "Gas Mask"
                break

        # ------------------------------
        # Push to memory
        # ------------------------------
        hist["helmet"].append(current_helmet)
        hist["mask"].append(current_mask)
        hist["gasmask"].append(current_gasmask)

        helmet_status = stable_vote(hist["helmet"], "No Helmet")
        mask_status = stable_vote(hist["mask"], "No Mask")
        gasmask_status = stable_vote(hist["gasmask"], "No Gas Mask")

        is_safe_now = (
            helmet_status == "Helmet" and
            (mask_status == "With Mask" or gasmask_status == "Gas Mask")
        )

        hist["safe"].append("SAFE" if is_safe_now else "UNSAFE")
        overall_status = stable_vote(hist["safe"], "UNSAFE")

        if overall_status == "SAFE":
            color = (0, 255, 0)
            safe_count += 1
        else:
            color = (0, 0, 255)
            unsafe_count += 1

        # ------------------------------
        # DRAW THIS PERSON ONLY
        # ------------------------------
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 3)

        label_x = px1
        label_y = max(30, py1 - 80)

        cv2.putText(output, overall_status, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        cv2.putText(output, helmet_status, (label_x, label_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(output, mask_status, (label_x, label_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(output, gasmask_status, (label_x, label_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ==============================
    # GLOBAL STATUS
    # ==============================
    total_persons = len(final_persons)

    if unsafe_count > 0:
        global_status = "❌ UNSAFE"
        global_color = (0, 0, 255)
    elif safe_count > 0:
        global_status = "✅ SAFE"
        global_color = (0, 255, 0)
    else:
        global_status = "⚠ NO PERSON"
        global_color = (0, 165, 255)

    cv2.putText(output, global_status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, global_color, 3)

    stats_text = f"Persons: {total_persons} | Safe: {safe_count} | Unsafe: {unsafe_count}"
    cv2.putText(output, stats_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    frame = cv2.resize(frame, (640, 480))
    output = process_frame(frame)

    cv2.imshow("Final PPE Detection - Multi Person", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed")