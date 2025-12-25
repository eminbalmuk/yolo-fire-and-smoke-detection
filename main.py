from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset.yaml"


def ensure_data_yaml(path: Path = DATA_YAML) -> Path:
    if path.exists():
        return path

    path.write_text(
        """# Auto-generated minimal data config
path: .
train: images/train/images
val: images/test/images
test: images/test/images
names:
  0: fire
  1: smoke
""",
        encoding="utf-8",
    )
    return path


def format_timecode(seconds: float, fps: float) -> str:
    td = dt.timedelta(seconds=seconds)
    ms = int((td.microseconds / 1_000_000) * 1000)
    return f"{td.seconds//3600:02d}:{(td.seconds//60)%60:02d}:{td.seconds%60:02d}.{ms:03d}"


def train_model(
    model_name: str,
    data: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: Optional[str],
    project: str,
    name: str,
    workers: int,
) -> None:

    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        workers=workers,
    )


def run_video_inference(
    weights: Path,
    source: Path,
    conf: float,
    iou: float,
    device: Optional[str],
    imgsz: int,
    classes: Optional[Iterable[str]],
    save_video: bool,
    output_path: Path,
    save_json: Optional[Path],
) -> List[dict]:

    model = YOLO(str(weights))
    class_names = {idx: name for idx, name in model.names.items()}

    if 0 in class_names and 1 in class_names:
            temp_name = class_names[0] 
            
            class_names[0] = class_names[1]
            
            class_names[1] = temp_name
            
            print(f"Sınıf ID 0->{class_names[0]}, ID 1->{class_names[1]}")

    class_filter = {c.lower() for c in classes} if classes else None

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detections: List[dict] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls)
                cls_name = class_names.get(cls_id, str(cls_id))
                if class_filter and cls_name.lower() not in class_filter:
                    continue
                score = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                t_seconds = frame_idx / fps
                detections.append(
                    {
                        "frame": frame_idx,
                        "time_seconds": round(t_seconds, 3),
                        "timecode": format_timecode(t_seconds, fps),
                        "class": cls_name,
                        "confidence": round(score, 3),
                        "bbox_xyxy": xyxy.tolist(),
                    }
                )

                if writer is not None:
                    x1, y1, x2, y2 = xyxy.tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{cls_name} {score:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

        if writer is not None:
            timestamp = format_timecode(frame_idx / fps, fps)
            text_x = width - 250  # sağa 250 piksel içeriden
            text_y = height - 25  # alttan 25 piksel yukarıdan

            cv2.putText(
                frame,
                f"t={timestamp}",
                (text_x, text_y), # koordinatlar
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if save_json:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    return detections


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test YOLOv11 on DFireDataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train YOLO")
    train_p.add_argument("--model", default="yolo11n.pt", help="Base model or checkpoint")
    train_p.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--batch", type=int, default=16)
    train_p.add_argument("--device", default=None, help="cpu or cuda:0")
    train_p.add_argument("--project", default="runs/detect", help="Ultralytics project dir")
    train_p.add_argument("--name", default="dfire-yolo", help="Run name")
    train_p.add_argument("--workers", type=int, default=8)

    detect_p = subparsers.add_parser("detect", help="Run video inference")
    detect_p.add_argument("--weights", default="runs/detect/dfire-yolo/weights/best.pt")
    detect_p.add_argument("--source", default="FP1.mp4")
    detect_p.add_argument("--conf", type=float, default=0.32)
    detect_p.add_argument("--iou", type=float, default=0.3)
    detect_p.add_argument("--imgsz", type=int, default=640)
    detect_p.add_argument("--device", default=None)
    detect_p.add_argument("--classes", nargs="*", default=["fire", "smoke"], help="Class names to keep")
    detect_p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    detect_p.add_argument("--output", default="runs/pred/FP1_annotated.mp4")
    detect_p.add_argument("--log-json", default="runs/pred/FP1_detections.json")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "train":
        data_yaml = ensure_data_yaml(Path(args.data))
        train_model(
            model_name=args.model,
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            workers=args.workers,
        )
    elif args.command == "detect":
        weights = Path(args.weights)
        source = Path(args.source)
        output_path = Path("predicted_videos") / f"{source.stem}_prediction.mp4"
        log_json = Path("predicted_videos_labels") / f"{source.stem}_predictionlabel.json" if args.log_json else None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if log_json:
            log_json.parent.mkdir(parents=True, exist_ok=True)

        detections = run_video_inference(
            weights=weights,
            source=source,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            classes=args.classes,
            save_video=args.save_video,
            output_path=output_path,
            save_json=log_json,
        )

        if detections:
            first_hit = min(detections, key=lambda d: d["time_seconds"])
            print(f"First detection at {first_hit['timecode']} ({first_hit['class']} conf={first_hit['confidence']})")
            print(f"Total detections: {len(detections)}")
            print(f"Log saved to {log_json}")
        else:
            print("No detections in the video.")


if __name__ == "__main__":
    main()
