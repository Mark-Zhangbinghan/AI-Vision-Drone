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
  5. 合并输出: bbox + 粗类 + 细类 + 置信度

Usage:
    python pipeline_inference.py --source test_image.jpg
    python pipeline_inference.py --source video.mp4
    python pipeline_inference.py --source D:/path/to/images/
    python pipeline_inference.py --source 0  # 摄像头
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
# 0=person_in_water(水中人, 送Stage2细分), 1=person(岸上人/安全, 直接输出)
STAGE1_CLASS_NAMES = ["person_in_water", "person"]

# Stage 2 类别 (nc=2)
STAGE2_CLASS_NAMES = ["drowning", "swimming"]

# 溺水警告阈值 (疑似阈值)
# drowning_conf >= 此值时, 报 drowning_possible (橙色警告)
DROWNING_THRESHOLD = 0.5

# 溺水确认阈值 (红色确认)
# 需 drowning_conf >= 此值 且 drowning_conf > swimming_conf 才报 drowning (红)
# 高于疑似阈值, 用于压制杂物/边界 case 的误报红框
DROWNING_CONFIRM = 0.65

# Stage2 最小分类置信度
# max(drowning_conf, swimming_conf) < 此值时, 视为"不确定/存疑",
# 不触发任何溺水告警 (中性框)。杂物 crop 通常两类概率都很低 → 落入此区间,
# 从而不再被强制判成 drowning/swimming 误报。
MIN_CLASS_CONF = 0.60

# Stage1 送 Stage2 的最低置信度闸门
# Stage1 单类检测器对杂物也会出框, 但多数低置信。
# conf < 此值时不再裁剪送 Stage2, 直接标中性 (person_in_water 未分类),
# 从源头砍掉一大批低质误检, 且不消耗 Stage2 算力。
ROUTE_CONF = 0.35

# 裁剪参数
CROP_PADDING = 0.2   # bbox 四周扩展比例
CROP_RESIZE = 256    # 裁剪后 resize 尺寸

class DrowningTracker:
    def __init__(self, window_size=60, alarm_ratio=0.6, stale_frame_threshold=60):
        self.window_size = window_size
        self.alarm_ratio = alarm_ratio
        self.stale_frame_threshold = stale_frame_threshold
        self.history = {}  # {track_id: {"history": deque, "last_seen": frame_index}}

    def update(self, track_id, is_drowning, frame_index=None):
        if track_id not in self.history:
            self.history[track_id] = {
                "history": deque(maxlen=self.window_size),
                "last_seen": frame_index if frame_index is not None else 0,
            }

        self.history[track_id]["history"].append(1 if is_drowning else 0)

        if frame_index is not None:
            self.history[track_id]["last_seen"] = frame_index

        drowning_count = sum(self.history[track_id]["history"])
        current_ratio = drowning_count / len(self.history[track_id]["history"])

        if current_ratio >= self.alarm_ratio:
            return "drowning"
        elif drowning_count > 0:
            return "drowning_possible"
        return "swimming"

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
    box_xyxy: [x1, y1, x2, y2] 绝对像素坐标
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    # 计算 padding
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    # 扩展 bbox (确保在图片范围内)
    x1 = max(0, int(x1) - pad_w)
    y1 = max(0, int(y1) - pad_h)
    x2 = min(img_w, int(x2) + pad_w)
    y2 = min(img_h, int(y2) + pad_h)

    # 裁剪
    crop = image[y1:y2, x1:x2]

    # 确保最小尺寸
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0, cx - 64)
        y1 = max(0, cy - 64)
        x2 = min(img_w, cx + 64)
        y2 = min(img_h, cy + 64)
        crop = image[y1:y2, x1:x2]

    # Resize
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

    Args:
        image: numpy array (BGR)
        stage1_model: YOLO detect model
        stage2_model: YOLO classify model
        drowning_threshold: drowning警告阈值 (drowning_conf > 此值但 < swimming_conf时报drowning_possible)

    Returns:
        results: list of dicts, each containing:
            - bbox: [x1, y1, x2, y2]
            - coarse_class: Stage 1 类名
            - fine_class: Stage 2 类名 (drowning/drowning_possible/swimming)
            - coarse_conf: Stage 1 置信度
            - fine_conf: Stage 2 置信度 (仅 person_in_water)
            - drowning_conf: drowning 置信度
            - swimming_conf: swimming 置信度
    """
    # Stage 1: 检测所有目标
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

        # 对 person_in_water 送入 Stage 2 分类
        # Stage1 nc=2: cls_id 0=person_in_water, 1=person
        # 用类名判断确保兼容 (不可依赖固定 cls_id)
        if coarse_class == "person_in_water":
            # Stage 2 未加载 → 仅输出粗粒度检测（中性框，提示需 Stage2 细分）
            if stage2_model is None:
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "coarse_conf": conf,
                    "fine_conf": conf,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
                continue

            # [阈值缓解] Stage1 低置信闸门:
            # 单类 Stage1 对杂物也会出框, 多数低置信。
            # conf 低于 route_conf 的框不再送 Stage2, 直接标中性,
            # 从源头砍掉一大批低质误检, 且省 Stage2 算力。
            if conf < route_conf:
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "coarse_conf": conf,
                    "fine_conf": conf,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
                continue

            # 裁剪 bbox 区域
            crop = crop_person_in_water(image, xyxy, CROP_PADDING)

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                # Stage 2: 分类
                stage2_result = stage2_model(crop, verbose=False)
                probs = stage2_result[0].probs

                drowning_conf = float(probs.data[0])  # drowning = class 0
                swimming_conf = float(probs.data[1])   # swimming = class 1

                # [阈值缓解] 分级 + 拒识别判定:
                # Stage2 是强迫二选一 (drowning/swimming), 杂物 crop 常被随机判成
                # 其中一类 → 误报红框。引入 "不确定" 档: 当两类最大概率都
                # 很低 (max < min_class_conf) 时, 视为存疑/非人, 不触发任何告警。
                max_conf = max(drowning_conf, swimming_conf)

                if max_conf < min_class_conf:
                    # 不确定 / 存疑: 杂物大概率落此区间, 中性框, 不告警
                    fine_class = "person_in_water(未分类)"
                    fine_conf = max_conf
                else:
                    is_drowning_now = (
                        drowning_conf > swimming_conf and
                        drowning_conf >= drowning_confirm
                    )

                    if tracker is not None and getattr(box, 'id', None) is not None:
                        try:
                            track_id = int(box.id[0])
                        except Exception:
                            track_id = None

                        if track_id is not None:
                            fine_class = tracker.update(
                                track_id,
                                is_drowning_now,
                                frame_index=frame_index,
                            )
                        else:
                            if is_drowning_now:
                                fine_class = "drowning"
                            elif drowning_conf >= drowning_threshold:
                                fine_class = "drowning_possible"
                            else:
                                fine_class = "swimming"
                    else:
                        if is_drowning_now:
                            fine_class = "drowning"
                        elif drowning_conf >= drowning_threshold:
                            fine_class = "drowning_possible"
                        else:
                            fine_class = "swimming"

                    fine_conf = drowning_conf if fine_class == "drowning" else swimming_conf

                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": fine_class,
                    "coarse_conf": conf,
                    "fine_conf": fine_conf,
                    "drowning_conf": drowning_conf,
                    "swimming_conf": swimming_conf,
                    "track_id": track_id,
                })
            else:
                # 裁剪失败, 只保留粗类
                output.append({
                    "bbox": xyxy.tolist(),
                    "coarse_class": coarse_class,
                    "fine_class": "person_in_water(未分类)",
                    "coarse_conf": conf,
                    "fine_conf": None,
                    "drowning_conf": None,
                    "swimming_conf": None,
                    "track_id": track_id,
                })
        else:
            # 非 person_in_water, 只保留粗类
            output.append({
                "bbox": xyxy.tolist(),
                "coarse_class": coarse_class,
                "fine_class": None,
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
    在图片上绘制两阶段推理结果

    颜色编码:
      - drowning: 红框 (高优先级告警)
      - drowning_possible: 橙框 (潜在溺水警告)
      - swimming: 绿框 (正常游泳)
      - person: 蓝框 (岸上/安全, 不告警)
      - person_in_water(未分类): 金框 (中性, 不告警)
    """
    colors = {
        "drowning": (0, 0, 255),              # BGR红色 - 确认溺水
        "drowning_possible": (0, 165, 255),    # BGR橙色 - 潜在溺水警告
        "swimming": (0, 255, 0),              # BGR绿色 - 正常游泳
        "person_in_water(未分类)": (0, 215, 255),  # BGR金色 - 中性未分类
        "person": (255, 0, 0),                # BGR蓝色 - 岸上人(安全)
    }

    img_out = image.copy()

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        fine_class = r.get("fine_class")

        # 选择颜色和标签
        if fine_class and fine_class in colors:
            color = colors[fine_class]
            if r.get("drowning_conf") is not None:
                label = f"{fine_class} D={r['drowning_conf']:.2f} S={r['swimming_conf']:.2f}"
            else:
                label = f"{fine_class} ({r['fine_conf']:.2f})"
        else:
            color = colors.get(r["coarse_class"], (128, 128, 128))
            label = f"{r['coarse_class']} ({r['coarse_conf']:.2f})"

        # 绘制 bbox
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, thickness)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 如果是 drowning, 画醒目的告警边框
        if fine_class == "drowning":
            cv2.rectangle(img_out, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                          (0, 0, 255), 3)
        elif fine_class == "drowning_possible":
            cv2.rectangle(img_out, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                          (0, 165, 255), 2)  # 橙色边框, 线宽2

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
                        help="Stage 1 模型权重路径 (nc=3: person_in_water/person/ignore)")
    parser.add_argument("--stage2-weights", type=str,
                        default=str(PROJECT_ROOT / "runs" / "surveil_stage2" /
                                    "yolo26s_cls_surveil_stage2_v1" / "weights" / "best.pt"),
                        help="Stage 2 模型权重路径 (nc=2: drowning/swimming)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Stage 1 检测置信度阈值 (默认0.35, 高于旧0.25以抑制杂物误检)")
    parser.add_argument("--drowning-threshold", type=float, default=DROWNING_THRESHOLD,
                        help="drowning 疑似阈值 (drowning_conf>=此值报 drowning_possible 橙框)")
    parser.add_argument("--drowning-confirm", type=float, default=DROWNING_CONFIRM,
                        help="drowning 确认阈值 (drowning_conf>=此值且>swimming_conf 才报 drowning 红框)")
    parser.add_argument("--min-class-conf", type=float, default=MIN_CLASS_CONF,
                        help="Stage2 最小分类置信度, max(d,s)<此值视为存疑/非人, 不告警")
    parser.add_argument("--route-conf", type=float, default=ROUTE_CONF,
                        help="Stage1 低于此置信度的框不送 Stage2 (源头砍低质误检)")
    parser.add_argument("--save", type=str, default=None,
                        help="保存结果图片/视频的路径")
    parser.add_argument("--show", action="store_true", default=False,
                        help="实时显示结果")
    args = parser.parse_args()

    # 使用 CLI 参数作为各阈值 (不修改全局变量, 传参给函数)
    cli_drowning_threshold = args.drowning_threshold
    cli_drowning_confirm = args.drowning_confirm
    cli_min_class_conf = args.min_class_conf
    cli_route_conf = args.route_conf

    # 加载模型
    from ultralytics import YOLO

    print(f"[推理] 加载 Stage 1 模型: {args.stage1_weights}")
    stage1 = YOLO(args.stage1_weights)
    stage1.conf = args.conf

    print(f"[推理] 加载 Stage 2 模型: {args.stage2_weights}")
    stage2 = YOLO(args.stage2_weights)

    tracker = DrowningTracker(window_size=30, alarm_ratio=0.6,
                              stale_frame_threshold=60)

    # 推理
    source = args.source

    # 判断源类型
    if source.isdigit():
        # 摄像头
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
        # 目录: 批量处理图片
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

            # 打印结果
            print(f"\n{img_path.name}:")
            for r in results:
                fine = r.get("fine_class", "")
                if fine:
                    print(f"  {fine}: conf={r['fine_conf']:.3f}, "
                          f"drowning={r['drowning_conf']:.3f}, "
                          f"swimming={r['swimming_conf']:.3f}")
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
        # 单张图片或视频
        image = cv2.imread(source)
        if image is not None:
            # 单张图片
            results = two_stage_inference(image, stage1, stage2,
                                          drowning_threshold=cli_drowning_threshold,
                                          route_conf=cli_route_conf,
                                          drowning_confirm=cli_drowning_confirm,
                                          min_class_conf=cli_min_class_conf)
            img_out = draw_results(image, results)

            print(f"\n推理结果:")
            for r in results:
                fine = r.get("fine_class", "")
                if fine:
                    print(f"  {fine}: conf={r['fine_conf']:.3f}")
                else:
                    print(f"  {r['coarse_class']}: conf={r['coarse_conf']:.3f}")

            if args.save:
                cv2.imwrite(args.save, img_out)
                print(f"\n保存到: {args.save}")

            if args.show:
                cv2.imshow("Two-Stage Inference", img_out)
                cv2.waitKey(0)
        else:
            # 视频
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

    print("\n推理完成!")


if __name__ == "__main__":
    main()
