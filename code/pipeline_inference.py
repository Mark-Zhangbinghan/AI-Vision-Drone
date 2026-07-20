"""
pipeline_inference.py - 两阶段统一推理脚本
=============================================
Stage 1 (YOLO detect, nc=2): 检测 → person_in_water / person(岸上)
Stage 2 (YOLO classify, nc=2): 细粒度分类 → drowning / swimming

推理流程:
  1. Stage 1 检测所有目标 (person_in_water/person, nc=2)
  2. 对 person_in_water bbox 裁剪区域送入 Stage 2 分类
  3. person 直接作为粗类输出, 不进 Stage 2
  4. 比较优先分类策略:
     drowning_conf > swimming_conf → drowning (确认溺水)
     drowning_conf > threshold 但 < swimming_conf → drowning_possible (潜在风险警告)
     drowning_conf <= threshold → swimming (正常游泳)
  5. 合并输出: bbox + 粗类 + 细类(纯模型结果) + 警报状态(滑动窗口结果)
"""

import sys
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# ===========================================================================
#  配置
# ===========================================================================

# Stage 1 类别 (nc=2, 监控视角)
STAGE1_CLASS_NAMES = ["person_in_water", "person"]

# Stage 2 类别 (nc=2)
STAGE2_CLASS_NAMES = ["drowning", "swimming"]

# 溺水警告阈值 (疑似阈值)
DROWNING_THRESHOLD = 0.5

# 溺水确认阈值 (红色确认)
DROWNING_CONFIRM = 0.65

# Stage2 最小分类置信度
MIN_CLASS_CONF = 0.60

# Stage1 送 Stage2 的最低置信度闸门
ROUTE_CONF = 0.35

# 裁剪参数
CROP_PADDING = 0.2   # bbox 四周扩展比例
CROP_RESIZE = 256    # 裁剪后 resize 尺寸

class DrowningTracker:
    def __init__(self, window_size=90, alarm_ratio=0.6, stale_frame_threshold=60):
        self.window_size = window_size
        self.alarm_ratio = alarm_ratio
        self.stale_frame_threshold = stale_frame_threshold
        self.history = {}  # {track_id: {"history": deque, "last_seen": frame_index}}

    def update(self, track_id, is_drowning_now, frame_index=None):
        """
        更新滑动窗口状态，并输出独立的滑动窗口警报结果 (alarm_class)。
        """
        if track_id not in self.history:
            self.history[track_id] = {
                "history": deque(maxlen=self.window_size),
                "last_seen": frame_index if frame_index is not None else 0,
            }

        self.history[track_id]["history"].append(1 if is_drowning_now else 0)

        if frame_index is not None:
            self.history[track_id]["last_seen"] = frame_index

        history_deque = self.history[track_id]["history"]
        drowning_count = sum(history_deque)

        # 使用滑动窗口长度计算比例
        current_ratio = drowning_count / self.window_size

        # 滑动窗口独立判断警报类别
        if current_ratio >= self.alarm_ratio:
            alarm_class = "drowning_alarm"
        elif drowning_count > 0:
            alarm_class = "drowning_possible"
        else:
            alarm_class = "swimming"

        return {
            "alarm_class": alarm_class,
            "is_alarm": current_ratio >= self.alarm_ratio,
            "drowning_ratio": current_ratio,
            "drowning_count": drowning_count
        }

    def cleanup(self, active_ids, current_frame_index):
        if active_ids is None or current_frame_index is None:
            return

        keys_to_delete = [
            tid for tid, data in self.history.items()
            if tid not in active_ids and
            (current_frame_index - data["last_seen"]) >= self.stale_frame_threshold
        ]

        for tid in keys_to_delete:
            del self.history[tid]

# ===========================================================================
#  裁剪函数
# ===========================================================================

def crop_person_in_water(image, box_xyxy, padding=CROP_PADDING):
    """
    从原图裁剪 person_in_water bbox 区域, 带 padding
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, int(x1) - pad_w)
    y1 = max(0, int(y1) - pad_h)
    x2 = min(img_w, int(x2) + pad_w)
    y2 = min(img_h, int(y2) + pad_h)

    crop = image[y1:y2, x1:x2]

    if crop.shape[0] < 32 or crop.shape[1] < 32:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0, cx - 64)
        y1 = max(0, cy - 64)
        x2 = min(img_w, cx + 64)
        y2 = min(img_h, cy + 64)
        crop = image[y1:y2, x1:x2]

    if CROP_RESIZE > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
        crop = cv2.resize(crop, (CROP_RESIZE, CROP_RESIZE),
                         interpolation=cv2.INTER_LINEAR)

    return crop


# ===========================================================================
#  两阶段推理
# ===========================================================================

def two_stage_inference(image, stage1_model, stage2_model=None,
                        tracker=None,
                        frame_index=None,
                        drowning_threshold=DROWNING_THRESHOLD,
                        route_conf=ROUTE_CONF,
                        drowning_confirm=DROWNING_CONFIRM,
                        min_class_conf=MIN_CLASS_CONF):
    """
    两阶段推理: Stage 1 检测 → Stage 2 分类
    """
    if tracker is not None:
        stage1_results = stage1_model.track(image,
                                            persist=True,
                                            verbose=False,
                                            tracker="bytetrack.yaml")
    else:
        stage1_results = stage1_model(image, verbose=False)

    if not stage1_results or len(stage1_results) == 0:
        return []

    result = stage1_results[0]
    boxes = result.boxes

    output = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        track_id = None
        if getattr(box, 'id', None) is not None:
            try:
                track_id = int(box.id[0])
            except Exception:
                track_id = None

        coarse_class = STAGE1_CLASS_NAMES[cls_id]

        if coarse_class == "person_in_water":
            if stage2_model is None:
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "alarm_class": "normal",
                    "is_alarm": False,
                    "coarse_conf": conf,
                    "fine_conf": conf,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
                continue

            if conf < route_conf:
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "alarm_class": "normal",
                    "is_alarm": False,
                    "coarse_conf": conf,
                    "fine_conf": conf,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
                continue

            crop = crop_person_in_water(image, xyxy, CROP_PADDING)

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                stage2_result = stage2_model(crop, verbose=False)
                probs = stage2_result[0].probs

                drowning_conf = float(probs.data[0])
                swimming_conf = float(probs.data[1])

                max_conf = max(drowning_conf, swimming_conf)

                if max_conf < min_class_conf:
                    fine_class = "person_in_water(未分类)"
                    fine_conf = max_conf
                    is_drowning_now = False
                else:
                    # 纯模型结果 (fine_class)
                    is_drowning_now = (
                        drowning_conf > swimming_conf and
                        drowning_conf >= drowning_confirm
                    )

                    if is_drowning_now:
                        fine_class = "drowning"
                    elif drowning_conf >= drowning_threshold:
                        fine_class = "drowning_possible"
                    else:
                        fine_class = "swimming"

                    fine_conf = drowning_conf if fine_class == "drowning" else swimming_conf

                # 独立通过滑动窗口判定警报类别 (alarm_class)
                alarm_class = "normal"
                is_alarm = False
                if tracker is not None and track_id is not None:
                    tracker_res = tracker.update(
                        track_id,
                        is_drowning_now,
                        frame_index=frame_index,
                    )
                    alarm_class = tracker_res["alarm_class"]
                    is_alarm = tracker_res["is_alarm"]
                else:
                    if is_drowning_now:
                        alarm_class = "drowning_alarm"
                        is_alarm = True
                    elif drowning_conf >= drowning_threshold:
                        alarm_class = "drowning_possible"

                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": fine_class,
                    "alarm_class": alarm_class,
                    "is_alarm": is_alarm,
                    "coarse_conf": conf,
                    "fine_conf": fine_conf if max_conf >= min_class_conf else max_conf,
                    "drowning_conf": drowning_conf,
                    "swimming_conf": swimming_conf,
                    "track_id": track_id,
                })
            else:
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "alarm_class": "normal",
                    "is_alarm": False,
                    "coarse_conf": conf,
                    "fine_conf": None,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
        else:
            output.append({
                "bbox": xyxy.tolist(),
                "coarse_class": coarse_class,
                "fine_class": None,
                "alarm_class": "normal",
                "is_alarm": False,
                "coarse_conf": conf,
                "fine_conf": None,
                "drowning_conf": None,
                "swimming_conf": None,
                "track_id": track_id,
            })

    return output


# ===========================================================================
#  可视化
# ===========================================================================

def draw_results(image, results):
    """
    根据模型分类结果以及滑动窗口的 alarm_class 呈现可视化。
    """
    # 基础模型分类的颜色映射
    colors = {
        "drowning": (0, 0, 255),              # 红色
        "drowning_possible": (0, 165, 255),    # 橙色
        "swimming": (0, 255, 0),              # 绿色
        "person_in_water(未分类)": (0, 215, 255),  # 金色
        "person": (255, 0, 0),                # 蓝色
    }

    img_out = image.copy()

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        fine_class = r.get("fine_class")
        alarm_class = r.get("alarm_class", "normal")
        is_alarm = r.get("is_alarm", False)

        if fine_class and fine_class in colors:
            color = colors[fine_class]
            if r.get("drowning_conf") is not None:
                label = f"{fine_class} | Alarm:{alarm_class} D={r['drowning_conf']:.2f}"
            else:
                label = f"{fine_class} ({r['fine_conf']:.2f})"
        else:
            color = colors.get(r["coarse_class"], (128, 128, 128))
            label = f"{r['coarse_class']} ({r['coarse_conf']:.2f})"

        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, thickness)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 根据滑动窗口独立返回的报警状态进行外边框强调
        if is_alarm or alarm_class == "drowning_alarm":
            cv2.rectangle(img_out, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                          (0, 0, 255), 3)
        elif alarm_class == "drowning_possible":
            cv2.rectangle(img_out, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                          (0, 165, 255), 2)

    return img_out


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="两阶段统一推理")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源: 图片路径/视频路径/目录/摄像头ID")
    parser.add_argument("--stage1-weights", type=str,
                        default=str(PROJECT_ROOT / "runs" / "surveil_stage1" /
                                    "yolo26s_surveil_stage1_v1" / "weights" / "best.pt"),
                        help="Stage 1 模型权重路径")
    parser.add_argument("--stage2-weights", type=str,
                        default=str(PROJECT_ROOT / "runs" / "surveil_stage2" /
                                    "yolo26s_cls_surveil_stage2_v1" / "weights" / "best.pt"),
                        help="Stage 2 模型权重路径")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Stage 1 检测置信度阈值")
    parser.add_argument("--drowning-threshold", type=float, default=DROWNING_THRESHOLD,
                        help="drowning 疑似阈值")
    parser.add_argument("--drowning-confirm", type=float, default=DROWNING_CONFIRM,
                        help="drowning 确认阈值")
    parser.add_argument("--min-class-conf", type=float, default=MIN_CLASS_CONF,
                        help="Stage2 最小分类置信度")
    parser.add_argument("--route-conf", type=float, default=ROUTE_CONF,
                        help="Stage1 低于此置信度的框不送 Stage2")
    parser.add_argument("--save", type=str, default=None,
                        help="保存结果图片/视频的路径")
    parser.add_argument("--show", action="store_true", default=False,
                        help="实时显示结果")
    args = parser.parse_args()

    cli_drowning_threshold = args.drowning_threshold
    cli_drowning_confirm = args.drowning_confirm
    cli_min_class_conf = args.min_class_conf
    cli_route_conf = args.route_conf

    from ultralytics import YOLO

    print(f"[推理] 加载 Stage 1 模型: {args.stage1_weights}")
    stage1 = YOLO(args.stage1_weights)
    stage1.conf = args.conf

    print(f"[推理] 加载 Stage 2 模型: {args.stage2_weights}")
    stage2 = YOLO(args.stage2_weights)

    tracker = DrowningTracker(window_size=90, alarm_ratio=0.6,
                              stale_frame_threshold=60)

    source = args.source

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = two_stage_inference(frame, stage1, stage2,
                                          tracker=tracker,
                                          frame_index=frame_index,
                                          drowning_threshold=cli_drowning_threshold,
                                          route_conf=cli_route_conf,
                                          drowning_confirm=cli_drowning_confirm,
                                          min_class_conf=cli_min_class_conf)
            img_out = draw_results(frame, results)

            active_ids = {r["track_id"] for r in results if r.get("track_id") is not None}
            tracker.cleanup(active_ids, frame_index)
            frame_index += 1

            if args.show:
                cv2.imshow("Two-Stage Inference", img_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    elif Path(source).is_dir():
        img_files = list(Path(source).glob("*"))
        img_files = [f for f in img_files if f.suffix.lower() in
                     (".jpg", ".jpeg", ".png", ".bmp")]

        print(f"[推理] 找到 {len(img_files)} 张图片")

        for img_path in img_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            results = two_stage_inference(image, stage1, stage2,
                                          drowning_threshold=cli_drowning_threshold,
                                          route_conf=cli_route_conf,
                                          drowning_confirm=cli_drowning_confirm,
                                          min_class_conf=cli_min_class_conf)
            img_out = draw_results(image, results)

            print(f"\n{img_path.name}:")
            for r in results:
                fine = r.get("fine_class", "")
                alarm = r.get("alarm_class", "")
                if fine:
                    print(f"  model_fine: {fine}, alarm_class: {alarm}, "
                          f"drowning={r['drowning_conf']:.3f}, swimming={r['swimming_conf']:.3f}")
                else:
                    print(f"  {r['coarse_class']}: conf={r['coarse_conf']:.3f}")

            if args.save:
                save_path = Path(args.save) / f"result_{img_path.name}"
                cv2.imwrite(str(save_path), img_out)

            if args.show:
                cv2.imshow("Two-Stage Inference", img_out)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    else:
        image = cv2.imread(source)
        if image is not None:
            results = two_stage_inference(image, stage1, stage2,
                                          drowning_threshold=cli_drowning_threshold,
                                          route_conf=cli_route_conf,
                                          drowning_confirm=cli_drowning_confirm,
                                          min_class_conf=cli_min_class_conf)
            img_out = draw_results(image, results)

            print(f"\n推理结果:")
            for r in results:
                fine = r.get("fine_class", "")
                alarm = r.get("alarm_class", "")
                if fine:
                    print(f"  model_fine: {fine}, alarm_class: {alarm}")
                else:
                    print(f"  {r['coarse_class']}: conf={r['coarse_conf']:.3f}")

            if args.save:
                cv2.imwrite(args.save, img_out)
                print(f"\n保存到: {args.save}")

            if args.show:
                cv2.imshow("Two-Stage Inference", img_out)
                cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = None
            if args.save:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = two_stage_inference(frame, stage1, stage2,
                                              tracker=tracker,
                                              frame_index=frame_index,
                                              drowning_threshold=cli_drowning_threshold,
                                              route_conf=cli_route_conf,
                                              drowning_confirm=cli_drowning_confirm,
                                              min_class_conf=cli_min_class_conf)
                img_out = draw_results(frame, results)

                active_ids = {r["track_id"] for r in results if r.get("track_id") is not None}
                tracker.cleanup(active_ids, frame_index)
                frame_index += 1

                if writer:
                    writer.write(img_out)

                if args.show:
                    cv2.imshow("Two-Stage Inference", img_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    print("\n推理完成！")


if __name__ == "__main__":
    main()