"""
Forma & PPE Compliance Core Module (from forma_dataset).
Dual-Model Fusion: Dedicated Person Detector + Fine-tuned Forma Detector (best.pt).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
FORMA_WEIGHTS = ROOT_DIR / "models" / "forma_best.pt"
PERSON_WEIGHTS = ROOT_DIR / "models" / "person_detector.pt"

def box_iou(b1, b2):
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter = max(0.0, xi_max - xi_min) * max(0.0, yi_max - yi_min)
    area1 = max(1.0, (x1_max - x1_min) * (y1_max - y1_min))
    area2 = max(1.0, (x2_max - x2_min) * (y2_max - y2_min))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def box_containment(b1, b2):
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter = max(0.0, xi_max - xi_min) * max(0.0, yi_max - yi_min)
    area1 = max(1.0, (x1_max - x1_min) * (y1_max - y1_min))
    area2 = max(1.0, (x2_max - x2_min) * (y2_max - y2_min))
    return inter / min(area1, area2)

def is_valid_human_box(box, conf, source="Forma"):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    ar = h / w
    if w < 10.0 or h < 18.0:
        return False
    # Sanoat kameralari va veb-kameralarda o'tirgan, yaqin turgan yoki uzoqdagi ishchilarni to'g'ri qabul qilish
    if ar < 0.40 or ar > 5.5:
        return False
    return True

def is_same_person(b1, b2):
    iou = box_iou(b1, b2)
    cont = box_containment(b1, b2)
    if iou >= 0.28 or cont >= 0.48:
        return True
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    cx1 = (b1[0] + b1[2]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    dx = abs(cx1 - cx2) / min(w1, w2)
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_h = max(0.0, yi2 - yi1)
    v_overlap = inter_h / min(h1, h2)
    if dx < 0.38 and v_overlap > 0.50:
        return True
    return False

def merge_person_boxes(b1, b2):
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3])
    ]

def deduplicate_and_merge_persons(candidates):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["conf"], reverse=True)
    merged = []
    for cand in candidates:
        matched = False
        for m in merged:
            if is_same_person(cand["box"], m["box"]):
                m["box"] = merge_person_boxes(cand["box"], m["box"])
                m["conf"] = max(m["conf"], cand["conf"])
                if cand.get("source") != m.get("source"):
                    m["source"] = "Dual Model"
                    m["conf"] = min(0.99, m["conf"] + 0.04)
                matched = True
                break
        if not matched:
            merged.append(dict(cand))

    changed = True
    while changed:
        changed = False
        new_merged = []
        skip_indices = set()
        for i in range(len(merged)):
            if i in skip_indices:
                continue
            curr = merged[i]
            for j in range(i + 1, len(merged)):
                if j in skip_indices:
                    continue
                if is_same_person(curr["box"], merged[j]["box"]):
                    curr["box"] = merge_person_boxes(curr["box"], merged[j]["box"])
                    curr["conf"] = max(curr["conf"], merged[j]["conf"])
                    curr["source"] = "Dual Model"
                    skip_indices.add(j)
                    changed = True
            new_merged.append(curr)
        merged = new_merged
    return merged

def is_stripe_strictly_inside_person(stripe_box, person_box):
    sx1, sy1, sx2, sy2 = stripe_box
    px1, py1, px2, py2 = person_box
    ph = max(1.0, py2 - py1)
    scx = (sx1 + sx2) / 2.0
    scy = (sy1 + sy2) / 2.0
    if not (px1 <= scx <= px2):
        return False
    if not (py1 + 0.05 * ph <= scy <= py2 - 0.02 * ph):
        return False
    ix1 = max(sx1, px1)
    iy1 = max(sy1, py1)
    ix2 = min(sx2, px2)
    iy2 = min(sy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    stripe_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    return (inter_area / stripe_area) >= 0.65


class FormaPPEPipeline:
    def __init__(self, forma_path: str = str(FORMA_WEIGHTS), person_path: str = str(PERSON_WEIGHTS), device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"📦 Forma modeli yuklanmoqda: {forma_path} ({self.device})...")
        self.forma_model = YOLO(forma_path)
        self.forma_model.to(self.device)
        self.person_model = None
        if Path(person_path).exists():
            print(f"📦 PersonGate modeli yuklanmoqda: {person_path} ({self.device})...")
            self.person_model = YOLO(person_path)
            self.person_model.to(self.device)

    def detect(self, frame: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> Dict[str, Any]:
        person_candidates = []
        raw_stripes = []

        # 1. Dedicated PersonGate Detector
        if self.person_model is not None:
            res_p = self.person_model.predict(
                frame,
                conf=max(0.15, conf * 0.75),
                imgsz=imgsz,
                classes=[0],
                verbose=False,
                device=self.device
            )[0]
            for b in res_p.boxes:
                xyxy = [float(v) for v in b.xyxy[0]]
                score = float(b.conf[0])
                if is_valid_human_box(xyxy, score, source="PersonGate"):
                    person_candidates.append({"box": xyxy, "conf": score, "source": "PersonGate"})

        # 2. Forma & Stripe Detector (best.pt)
        res_f = self.forma_model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )[0]
        for b in res_f.boxes:
            cls_id = int(b.cls[0])
            xyxy = [float(v) for v in b.xyxy[0]]
            score = float(b.conf[0])
            if cls_id == 0:  # person
                if is_valid_human_box(xyxy, score, source="Forma"):
                    person_candidates.append({"box": xyxy, "conf": score, "source": "Forma"})
            elif cls_id == 1:  # yaltiroq-chiziq
                raw_stripes.append(xyxy)

        # 3. Deduplicate persons
        persons = deduplicate_and_merge_persons(person_candidates)

        # 4. Filter stripes that are strictly inside persons
        valid_stripes = []
        for s_box in raw_stripes:
            for p_data in persons:
                if is_stripe_strictly_inside_person(s_box, p_data["box"]):
                    valid_stripes.append(s_box)
                    break

        # 5. Check compliance for each person
        formada_cnt = 0
        formasiz_cnt = 0
        for p in persons:
            p_box = p["box"]
            has_forma = False
            for s_box in valid_stripes:
                if is_stripe_strictly_inside_person(s_box, p_box):
                    has_forma = True
                    break
            p["has_forma"] = has_forma
            if has_forma:
                formada_cnt += 1
            else:
                formasiz_cnt += 1

        return {
            "persons": persons,
            "stripes": valid_stripes,
            "formada_count": formada_cnt,
            "formasiz_count": formasiz_cnt,
            "total_persons": len(persons)
        }
