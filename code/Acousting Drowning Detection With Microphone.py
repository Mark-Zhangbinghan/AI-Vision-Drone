import cv2
import numpy as np
import sounddevice as sd
import queue
import time
from collections import deque

# --- System Configuration ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
HISTORY_SIZE = 400

# --- AI Tuning Variables ---
MIN_VOLUME = 0.10          # Lower threshold to catch medium splashes
JITTER_THRESHOLD = 0.25    # Arrhythmia trigger
IDLE_TIMEOUT = 4.0
DEBOUNCE_TIME = 0.10       # Blocks immediate echoes but catches rapid thrashing
DROWNING_CONFIRM_SECONDS = 5  # Sustained early drowning → confirmed

# --- State Stickiness (Hysteresis) ---
REQUIRED_BAD_BEATS = 3     # How many erratic strokes after entry to trigger alert
REQUIRED_GOOD_BEATS = 4    # How many steady strokes to clear the alert

# --- Global drowning flag for external access (same interface as hydrophone_testing) ---
drowning = False
running = True   # set to False externally to trigger graceful shutdown

# --- Data Structures ---
q = queue.Queue()
energy_history = deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
jitter_history = deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

# Strict 4-stroke window
ibi_buffer = deque(maxlen=4)

# --- State Variables ---
last_splash_time = time.time()
stroke_armed = True

jitter_index = 0.0
display_pace = 0.00

# --- UI Dimensions ---
WIDTH, HEIGHT = 1000, 650
TOP_HEIGHT = 350


def callback(indata, frames, time_info, status):
    """sounddevice audio callback: push raw samples into the processing queue."""
    if status:
        print(f"[MIC] Audio callback status: {status}")
    q.put(indata.copy().flatten())


def main():
    """
    Start microphone-based drowning detection with live visualization window.
    Designed to run in a background daemon thread.
    Exposes the global 'drowning' flag for external polling
    (same interface as hydrophone_testing.main / hydrophone_testing.drowning).
    """
    global drowning, running, last_splash_time, stroke_armed, jitter_index, display_pace

    stream = sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE, dtype='float32')
    stream.start()

    print("[MIC] Microphone Variance Guard Active. Ready for testing...")

    # Per-thread state machine
    internal_state = "SILENT"
    system_status = "NO ONE IN WATER"
    status_color = (150, 150, 150)  # Gray
    consecutive_good_beats = 0
    consecutive_bad_beats = 0
    early_drowning_start_time = None

    try:
        while running:
            try:
                audio_data = q.get(timeout=0.5)
                current_time = time.time()

                # 1. LIVE AUDIO
                peak_energy = np.max(np.abs(audio_data))
                energy_history.append(peak_energy)

                # --- 2. PEAK & VALLEY STROKE DETECTION ---
                if peak_energy < MIN_VOLUME:
                    stroke_armed = True

                elif peak_energy > MIN_VOLUME and stroke_armed:
                    stroke_armed = False

                    interval = current_time - last_splash_time

                    # --- THE DEBOUNCE FILTER ---
                    if interval > DEBOUNCE_TIME:
                        ibi_buffer.append(interval)

                        # --- 3. PURE VARIANCE MATH ---
                        if len(ibi_buffer) >= 3:
                            live_mean = np.mean(ibi_buffer)
                            live_std = np.std(ibi_buffer)

                            jitter_index = live_std / (live_mean + 0.05)
                            display_pace = live_mean

                            # --- 4. HYSTERESIS STICKINESS & STATE MACHINE ---
                            if jitter_index > JITTER_THRESHOLD:
                                consecutive_bad_beats += 1
                                consecutive_good_beats = 0
                            else:
                                consecutive_good_beats += 1
                                consecutive_bad_beats = 0

                            # WATER ENTRY GRACE PERIOD
                            if internal_state == "SILENT":
                                internal_state = "SWIMMING"
                                system_status = "WATER ENTRY: CALIBRATING..."
                                status_color = (100, 255, 100)  # Green
                                print("[MIC] WATER ENTRY: CALIBRATING...")
                                consecutive_bad_beats = 0

                            # Require X consecutive BAD intervals to shift to Drowning
                            elif internal_state == "SWIMMING":
                                if consecutive_bad_beats >= REQUIRED_BAD_BEATS:
                                    internal_state = "EARLY_DROWNING"
                                    system_status = "EARLY STAGES OF DROWNING"
                                    status_color = (50, 50, 255)  # Red
                                    early_drowning_start_time = current_time
                                    print("[MIC] EARLY STAGES OF DROWNING")
                                else:
                                    system_status = "STABLE: SWIMMING"
                                    status_color = (100, 255, 100)  # Green

                            # Require Y consecutive GOOD intervals to shift back to Swimming
                            elif (internal_state == "EARLY_DROWNING" or internal_state == "ALERT") \
                                    and consecutive_good_beats >= REQUIRED_GOOD_BEATS:
                                internal_state = "SWIMMING"
                                system_status = "STABLE: SWIMMING"
                                status_color = (100, 255, 100)  # Green
                                early_drowning_start_time = None
                                drowning = False
                                print("[MIC] STABLE: SWIMMING")

                        last_splash_time = current_time

                jitter_history.append(jitter_index)

                # --- 5. THE SILENT GASP REFLEX & TIMEOUT LOGIC ---
                time_since_beat = current_time - last_splash_time
                if time_since_beat > IDLE_TIMEOUT:
                    if internal_state == "EARLY_DROWNING" or internal_state == "ALERT":
                        internal_state = "ALERT"
                        system_status = "DROWNING HIGH ALERT"
                        drowning = True
                        print("[MIC] DROWNING HIGH ALERT - drowning flag set!")
                    elif internal_state == "SWIMMING":
                        internal_state = "SILENT"
                        system_status = "NO ONE IN WATER"
                        status_color = (150, 150, 150)  # Gray
                        drowning = False
                        display_pace = 0.00
                        jitter_index = 0.0
                        ibi_buffer.clear()
                        consecutive_good_beats = 0
                        consecutive_bad_beats = 0
                        early_drowning_start_time = None

                # --- 6. 5-SECOND EARLY DROWNING CONFIRMATION ---
                if internal_state == "EARLY_DROWNING" and early_drowning_start_time is not None:
                    if current_time - early_drowning_start_time >= DROWNING_CONFIRM_SECONDS:
                        internal_state = "ALERT"
                        system_status = "DROWNING HIGH ALERT"
                        drowning = True
                        print("[MIC] DROWNING CONFIRMED (5s sustained) - drowning flag set!")

            except queue.Empty:
                pass

            # ================================================================
            # --- 7. UI RENDERING (always runs, even after drowning flag set) ---
            # ================================================================
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            WHITE = (255, 255, 255)
            DARK_GRAY = (20, 20, 20)
            BRIGHT_GREEN = (100, 255, 100)
            BRIGHT_RED = (50, 50, 255)
            CYAN = (255, 255, 0)
            YELLOW = (0, 255, 255)

            # TOP ZONE — background changes with state
            bg_color = DARK_GRAY
            if internal_state == "SILENT":
                bg_color = DARK_GRAY
            elif internal_state == "SWIMMING":
                bg_color = (20, 50, 20)
            else:  # EARLY_DROWNING or ALERT
                bg_color = status_color

            cv2.rectangle(frame, (0, 0), (WIDTH, TOP_HEIGHT), bg_color, -1)

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.3
            thickness = 3
            text_size = cv2.getTextSize(system_status, font, font_scale, thickness)[0]
            text_x = (WIDTH - text_size[0]) // 2
            text_y = (TOP_HEIGHT + text_size[1]) // 2

            # Adjust text color for readability based on background
            text_color = WHITE if internal_state != "SWIMMING" else BRIGHT_GREEN
            if system_status == "WATER ENTRY: CALIBRATING...":
                text_color = WHITE

            cv2.putText(frame, system_status, (text_x, text_y), font, font_scale, text_color, thickness)

            # BOTTOM ZONE
            cv2.rectangle(frame, (0, TOP_HEIGHT), (WIDTH, HEIGHT), (30, 30, 30), -1)
            cv2.putText(frame, "MACHINE LEARNING: MICROPHONE VARIANCE GUARD", (30, TOP_HEIGHT + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRIGHT_GREEN, 2)

            cv2.putText(frame, f"Established Pace: {display_pace:.2f}s", (30, TOP_HEIGHT + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            cv2.putText(frame, f"Relative Variance: {jitter_index:.2f}", (400, TOP_HEIGHT + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        BRIGHT_RED if jitter_index > JITTER_THRESHOLD else CYAN, 1)
            cv2.putText(frame, f"Arrhythmia Trigger: > {JITTER_THRESHOLD:.2f}", (400, TOP_HEIGHT + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            # LIVE DIAGNOSTICS OVERLAY
            peak_energy_check = energy_history[-1] if energy_history else 0.0
            vol_color = YELLOW if peak_energy_check > MIN_VOLUME else WHITE
            arm_status = "ARMED" if stroke_armed else "RECHARGING"
            cv2.putText(frame, f"Mic Vol: {peak_energy_check:.3f} | Threshold: {MIN_VOLUME} | State: {arm_status}",
                        (30, TOP_HEIGHT + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vol_color, 1)
            cv2.putText(frame, f"Good Beats: {consecutive_good_beats}/{REQUIRED_GOOD_BEATS} | "
                               f"Bad Beats: {consecutive_bad_beats}/{REQUIRED_BAD_BEATS}",
                        (400, TOP_HEIGHT + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            # GRAPH
            graph_x_start = 30
            graph_width = WIDTH - 60
            graph_y_bottom = HEIGHT - 20
            graph_height = 90
            step = graph_width / HISTORY_SIZE

            list_energy = list(energy_history)
            list_jitter = list(jitter_history)

            cv2.rectangle(frame, (graph_x_start, graph_y_bottom - graph_height),
                          (graph_x_start + graph_width, graph_y_bottom), (50, 50, 50), 1)
            cv2.putText(frame, "Live Audio (White) vs. Rhythm Jitter (Cyan)",
                        (graph_x_start, graph_y_bottom - graph_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            thresh_y = int(graph_y_bottom - (JITTER_THRESHOLD * graph_height))
            cv2.line(frame, (graph_x_start, thresh_y), (graph_x_start + graph_width, thresh_y),
                     (80, 80, 80), 1, cv2.LINE_AA)

            for i in range(1, len(list_energy)):
                x1 = int(graph_x_start + (i - 1) * step)
                x2 = int(graph_x_start + i * step)

                y1_energy = int(graph_y_bottom - (list_energy[i - 1] * graph_height * 3.0))
                y2_energy = int(graph_y_bottom - (list_energy[i] * graph_height * 3.0))
                cv2.line(frame,
                         (x1, np.clip(y1_energy, graph_y_bottom - graph_height, graph_y_bottom)),
                         (x2, np.clip(y2_energy, graph_y_bottom - graph_height, graph_y_bottom)),
                         WHITE, 1)

                y1_jitter = int(graph_y_bottom - (list_jitter[i - 1] * graph_height))
                y2_jitter = int(graph_y_bottom - (list_jitter[i] * graph_height))

                line_color = BRIGHT_RED if list_jitter[i] > JITTER_THRESHOLD else CYAN
                cv2.line(frame,
                         (x1, np.clip(y1_jitter, graph_y_bottom - graph_height, graph_y_bottom)),
                         (x2, np.clip(y2_jitter, graph_y_bottom - graph_height, graph_y_bottom)),
                         line_color, 2, cv2.LINE_AA)

            cv2.imshow("Buoy Intelligent Guard", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break

    except cv2.error:
        # window was destroyed externally (e.g. main thread cleanup)
        pass

    finally:
        stream.stop()
        stream.close()
        cv2.destroyWindow("Buoy Intelligent Guard")
        print("[MIC] Microphone detection stopped")


if __name__ == "__main__":
    main()
