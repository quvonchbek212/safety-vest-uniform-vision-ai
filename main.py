"""
Safety Vest & High-Visibility Uniform Compliance Video Runner
"""

import argparse
import cv2
import time
from core.forma_engine import FormaComplianceEngine

def main():
    parser = argparse.ArgumentParser(description="PPE Safety Vest & Uniform Compliance Runner")
    parser.add_argument("--source", default="0", help="Webcam index, video file, or RTSP stream")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    engine = FormaComplianceEngine()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    print(f"[*] Starting PPE compliance monitoring on: {src} (Press 'q' to quit)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = engine.predict_compliance(frame, conf=args.conf)
        
        # Draw persons
        for p in res["persons"]:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            is_compliant = p.get("has_forma", False)
            color = (0, 200, 0) if is_compliant else (0, 0, 230)
            status_text = "FORMADA (COMPLIANT)" if is_compliant else "FORMASIZ (VIOLATION)"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status_text} | {p['conf']:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw reflective stripes
        for s in res["stripes"]:
            sx1, sy1, sx2, sy2 = [int(v) for v in s]
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)

        # Dashboard banner
        banner = f"Workers: {res['total_persons']} | Compliant: {res['formada_count']} | Non-Compliant: {res['formasiz_count']}"
        cv2.putText(frame, banner, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("PPE Safety Vest Compliance AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
