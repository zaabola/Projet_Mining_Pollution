"""
Fish Behavior Tracking & Contamination Analysis
Extracted from script 41 for web integration.
"""

import cv2
import numpy as np
import math
from collections import defaultdict, deque

# Configuration
IOU_THRESHOLD = 0.45
IMGSZ = 416
ALPHA = 0.4
CONF_ALPHA = 0.3
TRACK_TIMEOUT = 15
REID_LOST_TIMEOUT = 300
REID_HIST_THRESH = 0.65
REID_POS_WEIGHT = 0.5
HISTORY_LENGTH = 30
SPEED_THRESHOLD = 10.0
ACCEL_THRESHOLD = 4.0
DIR_THRESHOLD = 40.0
VERDICT_BUFFER = 90  # 3 seconds @ 30fps

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

    if fish_a["hist"] is not None and fish_b["hist"] is not None:
        hist_sim = cv2.compareHist(fish_a["hist"], fish_b["hist"], cv2.HISTCMP_CORREL)
        score += max(0.0, hist_sim) * 4.0
        weights_total += 4.0

    dx = fish_a["cx"] - fish_b["cx"]
    dy = fish_a["cy"] - fish_b["cy"]
    dist = math.sqrt(dx**2 + dy**2)
    pos_sim = max(0.0, 1.0 - (dist / 300.0))
    score += pos_sim * REID_POS_WEIGHT
    weights_total += REID_POS_WEIGHT
    
    area_a = fish_a["w"] * fish_a["h"]
    area_b = fish_b["w"] * fish_b["h"]
    if max(area_a, area_b) > 0:
        size_sim = min(area_a, area_b) / max(area_a, area_b)
        score += size_sim * 2.0
        weights_total += 2.0
        
    ar_a = fish_a["w"] / max(1, fish_a["h"])
    ar_b = fish_b["w"] / max(1, fish_b["h"])
    ar_sim = min(ar_a, ar_b) / max(ar_a, ar_b)
    score += ar_sim * 1.5
    weights_total += 1.5

    return score / weights_total if weights_total > 0 else 0.0

def calculate_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0: return 0.0
    
    val = dot / (mag1 * mag2)
    val = max(min(val, 1.0), -1.0)
    return math.degrees(math.acos(val))

def get_color(track_id, status="normal"):
    if status == "stressed":
        return (0, 0, 255)
    np.random.seed(int(track_id) * 7 + 13)
    hue = np.random.randint(0, 50) + 70
    color_hsv = np.uint8([[[hue, 255, 230]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)


class FishTracker:
    def __init__(self):
        self.active_fish = {}
        self.lost_fish = {}
        self.id_remap = {}
        self.next_stable_id = 1
        self.smooth_tracks = {}
        self.behavior_history = defaultdict(lambda: {
            "positions": deque(maxlen=HISTORY_LENGTH),
            "status": "normal",
            "speed": 0.0,
            "prev_speed": 0.0,
            "direction_change": 0.0,
        })
        self.water_verdict = "SAFE"
        self.verdict_counter = 0
        self.frame_count = 0

    def get_stable_id(self, bytetrack_id, fish_sig):
        if bytetrack_id in self.id_remap: return self.id_remap[bytetrack_id]

        best_match_id, best_score = None, 0.0
        for lost_id, lost_sig in self.lost_fish.items():
            score = compare_fish(fish_sig, lost_sig)
            if score > best_score: best_score, best_match_id = score, lost_id

        if best_match_id is not None and best_score >= REID_HIST_THRESH:
            self.id_remap[bytetrack_id] = best_match_id
            del self.lost_fish[best_match_id]
            return best_match_id

        stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.id_remap[bytetrack_id] = stable_id
        return stable_id

    def process_frame(self, model, frame, conf_threshold):
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        use_half = torch.cuda.is_available()

        self.frame_count += 1
        output_frame = frame.copy()
        
        results = model.track(
            frame, imgsz=IMGSZ, conf=conf_threshold, iou=IOU_THRESHOLD,
            persist=True, tracker="bytetrack.yaml", verbose=False,
            device=device, half=use_half
        )

        boxes = results[0].boxes
        seen_stable_ids = set()
        stressed_count = 0

        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bt_id = int(box.id[0]) if box.id is not None else None

                if bt_id is None: continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                hist = compute_histogram(frame, [x1, y1, x2, y2])
                fish_sig = {"cx": cx, "cy": cy, "w": x2-x1, "h": y2-y1, "hist": hist}

                stable_id = self.get_stable_id(bt_id, fish_sig)
                seen_stable_ids.add(stable_id)
                self.active_fish[stable_id] = {**fish_sig, "last_seen": self.frame_count}

                # Behavior Analysis
                b_hist = self.behavior_history[stable_id]
                b_hist["positions"].append((cx, cy))
                
                pts = list(b_hist["positions"])
                speed = 0.0
                dir_change = 0.0
                
                if len(pts) >= 5:
                    dx = pts[-1][0] - pts[-5][0]
                    dy = pts[-1][1] - pts[-5][1]
                    speed = math.sqrt(dx**2 + dy**2) / 5.0
                    b_hist["speed"] = speed
                    
                if len(pts) >= 10:
                    p1, p2, p3 = pts[-10], pts[-5], pts[-1]
                    dir_change = calculate_angle(p1, p2, p3)
                    b_hist["direction_change"] = dir_change

                accel = speed - b_hist.get("prev_speed", 0.0)
                b_hist["prev_speed"] = speed

                straightness = 1.0
                l_path = 0.0
                d_straight = 0.0
                if len(pts) >= 15:
                    d_straight = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
                    l_path = sum(math.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2) for i in range(1, len(pts)))
                    if l_path > 5.0: straightness = d_straight / l_path
                    
                is_darting = accel > ACCEL_THRESHOLD and speed > 5.0
                is_swimming_randomly = (straightness < 0.65 and l_path > 15.0 and d_straight > 25.0)

                if speed > SPEED_THRESHOLD or (dir_change > DIR_THRESHOLD and speed > 4.0) or is_swimming_randomly or is_darting:
                    b_hist["status"] = "stressed"
                else:
                    if speed < SPEED_THRESHOLD * 0.7 and (dir_change < DIR_THRESHOLD * 0.7 or speed <= 5.0) and not is_swimming_randomly:
                        b_hist["status"] = "normal"
                        
                if b_hist["status"] == "stressed":
                    stressed_count += 1

                # Smoothing
                raw_box = [x1, y1, x2, y2]
                if stable_id in self.smooth_tracks:
                    prev = self.smooth_tracks[stable_id]
                    smooth_box = [ALPHA * raw_box[i] + (1 - ALPHA) * prev["box"][i] for i in range(4)]
                    smooth_conf = CONF_ALPHA * conf + (1 - CONF_ALPHA) * prev["conf"]
                else:
                    smooth_box, smooth_conf = raw_box, conf

                self.smooth_tracks[stable_id] = {"box": smooth_box, "conf": smooth_conf, "last_seen": self.frame_count}

                sx1, sy1, sx2, sy2 = map(int, smooth_box)
                color = get_color(stable_id, b_hist["status"])
                
                # Draw trajectory
                for i in range(1, len(pts)):
                    cv2.line(output_frame, (int(pts[i-1][0]), int(pts[i-1][1])), 
                                    (int(pts[i][0]), int(pts[i][1])), color, max(1, int(2 * i / len(pts))))

                cv2.rectangle(output_frame, (sx1, sy1), (sx2, sy2), color, 2)

                stable_conf = round(smooth_conf * 20) / 20
                label = f"Fish #{stable_id} {stable_conf:.0%} (Spd:{speed:.1f})"
                (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(output_frame, (sx1, sy1-lh-bl-4), (sx1+lw, sy1), color, -1)
                cv2.putText(output_frame, label, (sx1, sy1-bl-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Cleanup Lost
        for sid in list(self.active_fish.keys()):
            if sid not in seen_stable_ids:
                if self.frame_count - self.active_fish[sid]["last_seen"] > TRACK_TIMEOUT:
                    self.lost_fish[sid] = self.active_fish[sid]
                    del self.active_fish[sid]
                    stale_bt_ids = [bt for bt, st in self.id_remap.items() if st == sid]
                    for bt in stale_bt_ids: del self.id_remap[bt]

        for lid in list(self.lost_fish.keys()):
            if self.frame_count - self.lost_fish[lid]["last_seen"] > REID_LOST_TIMEOUT:
                del self.lost_fish[lid]

        # Water Verdict
        active_tracked = len(seen_stable_ids)
        if active_tracked > 0:
            stressed_ratio = stressed_count / active_tracked
            is_contaminated = stressed_ratio >= 0.35
            
            if is_contaminated and self.water_verdict == "SAFE":
                self.verdict_counter += 1
                if self.verdict_counter > VERDICT_BUFFER:
                    self.water_verdict = "CONTAMINATED"
                    self.verdict_counter = 0
            elif not is_contaminated and self.water_verdict == "CONTAMINATED":
                self.verdict_counter += 1
                if self.verdict_counter > VERDICT_BUFFER:
                    self.water_verdict = "SAFE"
                    self.verdict_counter = 0
            else:
                self.verdict_counter = max(0, self.verdict_counter - 1)
                
        # UI Overlays
        h, w = output_frame.shape[:2]
        bg_color = (0, 0, 255) if self.water_verdict == "CONTAMINATED" else (0, 255, 0)
        text_color = (255, 255, 255) if self.water_verdict == "CONTAMINATED" else (0, 0, 0)
        cv2.rectangle(output_frame, (0, 0), (w, 50), bg_color, -1)
        cv2.putText(output_frame, f"WATER QUALITY: {self.water_verdict}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
        cv2.putText(output_frame, f"Stressed Fish: {stressed_count}/{active_tracked}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return output_frame
