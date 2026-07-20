"""
test_camera.py
=========================
摄像头实时测试:
1. 打开USB摄像头
2. Stage1 YOLO检测 person_in_water/person
3. Stage2 YOLO分类 drowning/swimming (纯模型结果 fine_class)
4. 结合 ByteTrack 与滑动窗口状态判定 (报警结果 alarm_class / is_alarm)
5. 实时显示检测与告警结果
"""

import cv2
from pathlib import Path

from ultralytics import YOLO

from pipeline_inference import (
    two_stage_inference,
    draw_results,
    DrowningTracker
)

# ==============================
# 模型路径
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent

STAGE1_WEIGHTS = (PROJECT_ROOT / "runs" / "yolo26s_surveil_stage1_v2" / "best.pt")

STAGE2_WEIGHTS = (PROJECT_ROOT / "runs" / "yolo26s_cls_surveil_stage2_v2" / "best.pt")

# ==============================
# 参数
# ==============================
CAMERA_ID = 0

STAGE1_CONF = 0.35

DROWNING_THRESHOLD = 0.5
DROWNING_CONFIRM = 0.65
MIN_CLASS_CONF = 0.60
ROUTE_CONF = 0.35


def main():

    print("[INFO] 加载 Stage1 模型...")
    stage1 = YOLO(str(STAGE1_WEIGHTS))

    print("[INFO] 加载 Stage2 模型...")
    stage2 = YOLO(str(STAGE2_WEIGHTS))

    stage1.conf = STAGE1_CONF

    # ==============================
    # ByteTrack & 滑动窗口状态初始化
    # ==============================
    tracker = DrowningTracker(
        window_size=90,
        alarm_ratio=0.6,
        stale_frame_threshold=60
    )

    # ==============================
    # 打开摄像头
    # ==============================
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("[ERROR] 摄像头打开失败")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] 摄像头启动")
    print("[INFO] 按 q 退出")

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[ERROR] 摄像头读取失败")
            break

        # ==============================
        # 两阶段推理 (返回 fine_class + alarm_class)
        # ==============================
        results = two_stage_inference(
            frame,
            stage1,
            stage2,
            tracker=tracker,
            frame_index=frame_id,
            drowning_threshold=DROWNING_THRESHOLD,
            drowning_confirm=DROWNING_CONFIRM,
            min_class_conf=MIN_CLASS_CONF,
            route_conf=ROUTE_CONF
        )

        # ==============================
        # 清理过期 track
        # ==============================
        active_ids = {
            r["track_id"]
            for r in results
            if r.get("track_id") is not None
        }

        tracker.cleanup(
            active_ids,
            frame_id
        )

        # ==============================
        # 绘制结果
        # ==============================
        output = draw_results(
            frame,
            results
        )

        # 改良：检查当前帧的所有目标中，是否有任意一个目标的滑动窗口触发了真实警报 (is_alarm)
        has_active_alarm = any(r.get("is_alarm", False) for r in results)

        if has_active_alarm:
            cv2.putText(
                output,
                "!!! DROWNING ALERT !!!",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

        # 显示帧率/帧序号
        cv2.putText(
            output,
            f"Frame: {frame_id}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Drowning Detection Camera",
            output
        )

        frame_id += 1

        # q 键退出
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()