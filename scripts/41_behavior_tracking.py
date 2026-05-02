"""
Fish Behavior Tracking & Re-Identification.

Features:
1. Re-Identification (ReID): Remaps ByteTrack IDs to persistent stable IDs when fish leave/return.
2. Behavior Analysis: Calculates speed, acceleration, and direction changes using a history window.
3. Contamination Detection: Flags erratic/fast swimming (indicative of phosphogypsum contamination).
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
import multiprocessing
import math
from collections import defaultdict, deque

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # ─────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMGSZ = 416              # match training resolution

    # Smoothing parameters
    ALPHA = 0.4               # EMA factor for bounding box
    CONF_ALPHA = 0.3          # Confidence smoothing factor
    TRACK_TIMEOUT = 15        # Frames to keep showing a track after it disappears

    # ── Re-ID parameters ──
    REID_LOST_TIMEOUT = 300   # Frames to remember a lost fish (10 sec @ 30fps) - handles long loops behind tank props
    REID_HIST_THRESH = 0.65   # Overall similarity threshold (increased since we have more features)
    REID_POS_WEIGHT = 0.5     # Significantly reduce position importance since they loop around the tank
    
    # ── Behavior & Contamination parameters ──
    HISTORY_LENGTH = 30       # Frames of history to keep per fish (1 second)
    SPEED_THRESHOLD = 10.0    # Lowered: Pixels per frame.
    ACCEL_THRESHOLD = 4.0     # Sudden darting/burst of speed
    DIR_THRESHOLD = 40.0      # Lowered: Direction change (degrees)
    
    # ─────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────
    MODEL_PATHS = [
        "runs/detect/runs/detect/models/yolo/yolov8_fish_v34/weights/best.pt",
        "runs/detect/models/yolo/yolov8_fish_v3/weights/best.pt",
        "runs/detect/models/yolo/yolov8m_fish_v2/weights/best.pt",
        "runs/detect/models/yolo/yolov8s_fish_merged/weights/best.pt",
    ]

    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("ERROR: No model found.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ─────────────────────────────────────────────
    # Video path
    # ─────────────────────────────────────────────
    video_path = "datasets/fish_dataset/test/images/nininini.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # ─────────────────────────────────────────────
    # Output setup
    # ─────────────────────────────────────────────
    os.makedirs("outputs/videos", exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    VERDICT_BUFFER = int(fps * 3) # Requires 3 seconds of sustained behavior to switch verdict

    output_path = "outputs/videos/fish_behavior.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # ─────────────────────────────────────────────
    # ReID Engine
    # ─────────────────────────────────────────────

    def compute_histogram(frame, box):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 5 or y2 - y1 < 5: return None

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def compare_fish(fish_a, fish_b):
        score = 0.0
        weights_total = 0.0

        # Feature 1: Color/Pattern Histogram (Highest Weight)
        if fish_a["hist"] is not None and fish_b["hist"] is not None:
            hist_sim = cv2.compareHist(fish_a["hist"], fish_b["hist"], cv2.HISTCMP_CORREL)
            score += max(0.0, hist_sim) * 4.0
            weights_total += 4.0

        # Feature 2: Position Proximity (Low weight, fish might teleport from left-to-right when looping)
        dx = fish_a["cx"] - fish_b["cx"]
        dy = fish_a["cy"] - fish_b["cy"]
        dist = math.sqrt(dx**2 + dy**2)
        # Score is 1.0 if close, drops to 0.0 if further than 300px
        pos_sim = max(0.0, 1.0 - (dist / 300.0))
        score += pos_sim * REID_POS_WEIGHT
        weights_total += REID_POS_WEIGHT
        
        # Feature 3: Size Similarity (Area)
        area_a = fish_a["w"] * fish_a["h"]
        area_b = fish_b["w"] * fish_b["h"]
        if max(area_a, area_b) > 0:
            size_sim = min(area_a, area_b) / max(area_a, area_b)
            score += size_sim * 2.0
            weights_total += 2.0
            
        # Feature 4: Aspect Ratio (Length vs Height)
        ar_a = fish_a["w"] / max(1, fish_a["h"])
        ar_b = fish_b["w"] / max(1, fish_b["h"])
        ar_sim = min(ar_a, ar_b) / max(ar_a, ar_b)
        score += ar_sim * 1.5
        weights_total += 1.5

        return score / weights_total if weights_total > 0 else 0.0

    # ─── Tracking State ───
    active_fish = {}
    lost_fish = {}
    id_remap = {}
    next_stable_id = 1
    smooth_tracks = {}
    
    # ─── Behavior State ───
    # stable_id -> { "positions": deque, "status": "normal" | "stressed" }
    behavior_history = defaultdict(lambda: {
        "positions": deque(maxlen=HISTORY_LENGTH),
        "status": "normal",
        "speed": 0.0,
        "prev_speed": 0.0,
        "direction_change": 0.0,
    })
    
    # Overall water verdict
    water_verdict = "SAFE"
    verdict_counter = 0

    def get_stable_id(bytetrack_id, fish_sig, frame_num):
        global next_stable_id
        if bytetrack_id in id_remap: return id_remap[bytetrack_id]

        best_match_id, best_score = None, 0.0
        for lost_id, lost_sig in lost_fish.items():
            score = compare_fish(fish_sig, lost_sig)
            if score > best_score: best_score, best_match_id = score, lost_id

        if best_match_id is not None and best_score >= REID_HIST_THRESH:
            id_remap[bytetrack_id] = best_match_id
            del lost_fish[best_match_id]
            return best_match_id

        stable_id = next_stable_id
        next_stable_id += 1
        id_remap[bytetrack_id] = stable_id
        return stable_id

    def get_color(track_id, status="normal"):
        if status == "stressed":
            # Red color for stressed fish
            return (0, 0, 255)
            
        np.random.seed(int(track_id) * 7 + 13)
        hue = np.random.randint(0, 50) + 70 # ensure distinct from red (0,0,255)
        color_hsv = np.uint8([[[hue, 255, 230]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)

    def calculate_angle(p1, p2, p3):
        """Calculate angle between 3 points (p1-p2 and p2-p3)"""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0: return 0.0
        
        val = dot / (mag1 * mag2)
        val = max(min(val, 1.0), -1.0) # Clamp between -1 and 1
        return math.degrees(math.acos(val))

    # ─────────────────────────────────────────────
    # Process video
    # ─────────────────────────────────────────────
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        results = model.track(
            frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )

        boxes = results[0].boxes
        seen_stable_ids = set()
        stressed_count = 0

        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bt_id = int(box.id[0]) if box.id is not None else None

            if bt_id is None: continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hist = compute_histogram(frame, [x1, y1, x2, y2])
            fish_sig = {"cx": cx, "cy": cy, "w": x2-x1, "h": y2-y1, "hist": hist}

            stable_id = get_stable_id(bt_id, fish_sig, frame_count)
            seen_stable_ids.add(stable_id)
            active_fish[stable_id] = {**fish_sig, "last_seen": frame_count}

            # ── Behavior Analysis ──
            b_hist = behavior_history[stable_id]
            b_hist["positions"].append((cx, cy))
            
            pts = list(b_hist["positions"])
            speed = 0.0
            dir_change = 0.0
            
            if len(pts) >= 5:
                # Calculate speed over last 5 frames
                dx = pts[-1][0] - pts[-5][0]
                dy = pts[-1][1] - pts[-5][1]
                speed = math.sqrt(dx**2 + dy**2) / 5.0
                b_hist["speed"] = speed
                
            if len(pts) >= 10:
                # Calculate direction change over last 10 frames
                p1, p2, p3 = pts[-10], pts[-5], pts[-1]
                dir_change = calculate_angle(p1, p2, p3)
                b_hist["direction_change"] = dir_change

            # Calculate continuous acceleration
            accel = speed - b_hist.get("prev_speed", 0.0)
            b_hist["prev_speed"] = speed

            # Path Straightness Index (to detect random/zigzag moving)
            straightness = 1.0
            l_path = 0.0
            if len(pts) >= 15:
                # Distance from start of array to end
                d_straight = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
                # Total actual path length traveled
                l_path = sum(math.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2) for i in range(1, len(pts)))
                if l_path > 5.0: straightness = d_straight / l_path
                
            # A fish is panicking if it makes SUDDEN moves or swims randomly (zigzags)!
            # (Ensuring it actually moved a chunk of distance 'd_straight > 25.0' and isn't just vibrating)
            is_darting = accel > ACCEL_THRESHOLD and speed > 5.0
            is_swimming_randomly = (straightness < 0.65 and l_path > 15.0 and d_straight > 25.0)

            # Classify behavior - HIGHLY SENSITIVE to panicking features
            if speed > SPEED_THRESHOLD or (dir_change > DIR_THRESHOLD and speed > 4.0) or is_swimming_randomly or is_darting:
                b_hist["status"] = "stressed"
            else:
                # Hysteresis to recover to normal
                if speed < SPEED_THRESHOLD * 0.7 and (dir_change < DIR_THRESHOLD * 0.7 or speed <= 5.0) and not is_swimming_randomly:
                     b_hist["status"] = "normal"
                     
            if b_hist["status"] == "stressed":
                stressed_count += 1

            # ── Smoothing & Drawing ──
            raw_box = [x1, y1, x2, y2]
            if stable_id in smooth_tracks:
                prev = smooth_tracks[stable_id]
                smooth_box = [ALPHA * raw_box[i] + (1 - ALPHA) * prev["box"][i] for i in range(4)]
                smooth_conf = CONF_ALPHA * conf + (1 - CONF_ALPHA) * prev["conf"]
            else:
                smooth_box, smooth_conf = raw_box, conf

            smooth_tracks[stable_id] = {"box": smooth_box, "conf": smooth_conf, "last_seen": frame_count}

            sx1, sy1, sx2, sy2 = map(int, smooth_box)
            color = get_color(stable_id, b_hist["status"])
            
            # Draw trajectory
            for i in range(1, len(pts)):
                cv2.line(frame, (int(pts[i-1][0]), int(pts[i-1][1])), 
                                (int(pts[i][0]), int(pts[i][1])), color, max(1, int(2 * i / len(pts))))

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

            stable_conf = round(smooth_conf * 20) / 20
            label = f"Fish #{stable_id} {stable_conf:.0%} (Spd:{speed:.1f})"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (sx1, sy1-lh-bl-4), (sx1+lw, sy1), color, -1)
            cv2.putText(frame, label, (sx1, sy1-bl-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # ── Cleanup Lost Fish ──
        for sid in list(active_fish.keys()):
            if sid not in seen_stable_ids:
                if frame_count - active_fish[sid]["last_seen"] > TRACK_TIMEOUT:
                    lost_fish[sid] = active_fish[sid]
                    del active_fish[sid]
                    stale_bt_ids = [bt for bt, st in id_remap.items() if st == sid]
                    for bt in stale_bt_ids: del id_remap[bt]

        for lid in list(lost_fish.keys()):
            if frame_count - lost_fish[lid]["last_seen"] > REID_LOST_TIMEOUT:
                del lost_fish[lid]

        # ── Water Verdict ──
        active_tracked = len(seen_stable_ids)
        if active_tracked > 0:
            stressed_ratio = stressed_count / active_tracked
            is_contaminated = stressed_ratio >= 0.35 # Heightened sensitivity (35% instead of 50%)
            
            if is_contaminated and water_verdict == "SAFE":
                verdict_counter += 1
                if verdict_counter > VERDICT_BUFFER:
                    water_verdict = "CONTAMINATED"
                    verdict_counter = 0
            elif not is_contaminated and water_verdict == "CONTAMINATED":
                verdict_counter += 1
                if verdict_counter > VERDICT_BUFFER:
                    water_verdict = "SAFE"
                    verdict_counter = 0
            else:
                verdict_counter = max(0, verdict_counter - 1)
                
        # ── UI Overlays ──
        # Verdict Banner
        bg_color = (0, 0, 255) if water_verdict == "CONTAMINATED" else (0, 255, 0)
        text_color = (255, 255, 255) if water_verdict == "CONTAMINATED" else (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (width, 50), bg_color, -1)
        cv2.putText(frame, f"WATER QUALITY: {water_verdict}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
        # Stats
        cv2.putText(frame, f"Stressed Fish: {stressed_count}/{active_tracked}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)

        cv2.imshow("Fish Behavior & Contamination Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if frame_count % 30 == 0:
             print(f"Frame {frame_count}/{total_frames} | Active: {active_tracked} | Stressed: {stressed_count} | Verdict: {water_verdict}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nSaved tracking output to: {output_path}")
