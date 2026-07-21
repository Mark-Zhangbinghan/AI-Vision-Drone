import pygame
import serial
import threading
import queue
import sys
import time
from collections import deque

# --- CONFIGURATION ---
PORT = 'COM5'
BAUD = 115200

# --- SENSOR TUNING CONTROLS ---
IMPACT_THRESHOLD = 20
DEBOUNCE_TIME = 0.12

# Global drowning flag for external access
# Becomes True if danger is confirmed and remains locked
drowning = False

# Thread-safe queue for serial data transfer
data_queue = queue.Queue()


def read_from_esp32(ser, q):
    """
    Background thread to continuously read serial data from ESP32.
    Parses and sends valid sensor data to the main thread via a queue.
    """
    while True:
        try:
            if ser.in_waiting > 0:
                raw_data = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_data:
                    q.put(raw_data)
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(1)


def main():
    global drowning

    print("ESP32-S3 Variance Guard Active. Ready for submerged testing...")

    # --- SERIAL PORT INITIALIZATION ---
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Connected to ESP32-S3 on {PORT}")
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
        print("Make sure the Arduino Serial Monitor is closed!")
        sys.exit(1)

    # Start serial communication thread
    serial_thread = threading.Thread(target=read_from_esp32, args=(ser, data_queue), daemon=True)
    serial_thread.start()

    # --- PYGAME GUI INITIALIZATION ---
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Buoy Intelligent Guard")

    # Font settings
    font_large = pygame.font.Font(None, 56)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 16)

    # Color palette
    BG_COLOR_TOP = (45, 45, 45)
    BG_COLOR_BOTTOM = (20, 20, 20)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    CYAN = (0, 255, 255)
    GRAY = (150, 150, 150)
    RED = (255, 50, 50)

    # Rhythm and status variables
    current_pace = 0.00
    rhythm_jitter = 0.00
    irregularity_trigger = 0.30

    # Store recent beat intervals to analyze swimming rhythm
    beat_intervals = deque(maxlen=4)
    last_beat_time = time.time()

    # Hysteresis counters to avoid flapping between states
    consecutive_good_beats = 0
    consecutive_bad_beats = 0

    # System state machine
    internal_state = "SILENT"
    system_status = "NO ONE IN WATER"
    status_color = GRAY

    # Timer for sustained early drowning detection
    early_drowning_start_time = None
    DROWNING_CONFIRM_SECONDS = 5

    # Scrolling history for real-time graph
    max_points = 250
    raw_history = deque([0] * max_points, maxlen=max_points)
    spike_history = deque([0] * max_points, maxlen=max_points)

    # Graph data decimation to reduce update frequency
    update_counter = 0
    current_max_raw = 0
    current_max_spike = 0

    running = True
    clock = pygame.time.Clock()

    # --- MAIN LOOP ---
    while running:
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Only update logic if drowning has not been confirmed
        if not drowning:
            try:
                # Process all available serial data
                while not data_queue.empty():
                    sensor_data = data_queue.get_nowait()

                    if "Raw:" in sensor_data and "VelocitySpike:" in sensor_data:
                        parts = sensor_data.split()

                        try:
                            # Extract raw sensor value
                            raw_str = [p for p in parts if "Raw:" in p][0]
                            raw_val = int(raw_str.split(":")[1])

                            # Extract velocity spike value
                            spike_str = [p for p in parts if "VelocitySpike:" in p][0]
                            spike_val = int(spike_str.split(":")[1])

                            # Accumulate peaks for graph downsampling
                            current_max_raw = max(current_max_raw, raw_val)
                            current_max_spike = max(current_max_spike, spike_val)
                            update_counter += 1

                            # Update graph every 10 samples to smooth display
                            if update_counter >= 10:
                                raw_history.append(current_max_raw)
                                spike_history.append(current_max_spike)
                                update_counter = 0
                                current_max_raw = 0
                                current_max_spike = 0

                            # Detect a valid physical impact or movement
                            if spike_val > IMPACT_THRESHOLD:
                                current_time = time.time()
                                interval = current_time - last_beat_time

                                # Debounce to prevent duplicate detections
                                if interval > DEBOUNCE_TIME:
                                    beat_intervals.append(interval)
                                    last_beat_time = current_time

                                    # Require at least 3 beats to establish rhythm
                                    if len(beat_intervals) >= 3:
                                        current_pace = sum(beat_intervals) / len(beat_intervals)
                                        variance = sum(abs(i - current_pace) for i in beat_intervals) / len(beat_intervals)
                                        rhythm_jitter = variance / current_pace

                                        # Classify beat as regular or irregular
                                        if rhythm_jitter > irregularity_trigger:
                                            consecutive_bad_beats += 1
                                            consecutive_good_beats = 0
                                        else:
                                            consecutive_good_beats += 1
                                            consecutive_bad_beats = 0

                                        # State transitions from SILENT (no activity)
                                        if internal_state == "SILENT":
                                            if rhythm_jitter > irregularity_trigger:
                                                internal_state = "EARLY_DROWNING"
                                                system_status = "EARLY STAGES OF DROWNING"
                                                status_color = RED
                                                early_drowning_start_time = time.time()
                                            else:
                                                internal_state = "SWIMMING"
                                                system_status = "STABLE: SWIMMING"
                                                status_color = GREEN
                                                early_drowning_start_time = None

                                        # Transition from stable swimming to early drowning
                                        elif internal_state == "SWIMMING" and consecutive_bad_beats >= 2:
                                            internal_state = "EARLY_DROWNING"
                                            system_status = "EARLY STAGES OF DROWNING"
                                            status_color = RED
                                            early_drowning_start_time = time.time()

                                        # Recovery from dangerous state to stable swimming
                                        elif (internal_state == "EARLY_DROWNING" or internal_state == "ALERT") and consecutive_good_beats >= 2:
                                            internal_state = "SWIMMING"
                                            system_status = "STABLE: SWIMMING"
                                            status_color = GREEN
                                            early_drowning_start_time = None

                        except (IndexError, ValueError):
                            pass

            except queue.Empty:
                pass

            # --- SILENT GASP REFLEX DETECTION ---
            time_since_beat = time.time() - last_beat_time
            if time_since_beat > 3.0:
                # Previously struggling and now silent → high alert
                if internal_state == "EARLY_DROWNING" or internal_state == "ALERT":
                    internal_state = "ALERT"
                    system_status = "DROWNING HIGH ALERT"
                # Was swimming then stopped → assume exited water
                elif internal_state == "SWIMMING":
                    internal_state = "SILENT"
                    system_status = "NO ONE IN WATER"
                    status_color = GRAY
                    current_pace = 0.00
                    rhythm_jitter = 0.00
                    beat_intervals.clear()
                    consecutive_good_beats = 0
                    consecutive_bad_beats = 0
                    early_drowning_start_time = None

            # --- 5-SECOND CONFIRMATION FOR DROWNING ---
            # Trigger drowning if early drowning persists for 5 seconds
            if internal_state == "EARLY_DROWNING" and early_drowning_start_time is not None:
                if time.time() - early_drowning_start_time >= DROWNING_CONFIRM_SECONDS:
                    drowning = True
                    system_status = "DROWNING CONFIRMED (5s WARNING)"
                    status_color = RED

            # Immediately confirm drowning if high alert state is reached
            if internal_state == "ALERT":
                drowning = True
                system_status = "DROWNING CONFIRMED (SILENT ALERT)"
                status_color = RED

        # Lock display to emergency state once drowning is confirmed
        if drowning:
            system_status = "!!! DROWNING DETECTED !!!"
            status_color = RED

        # Strobe effect for high alert and confirmed drowning
        display_color = status_color
        if internal_state == "ALERT" or drowning:
            if int(time.time() * 4) % 2 == 0:
                display_color = RED
            else:
                display_color = WHITE

        # --- DRAW UI BACKGROUND ---
        screen.fill(BG_COLOR_TOP)
        pygame.draw.rect(screen, BG_COLOR_BOTTOM, (0, HEIGHT // 2, WIDTH, HEIGHT // 2))

        # Title text
        title_text = font_small.render("Buoy Intelligent Guard", True, WHITE)
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 10))

        # Main status display
        status_render = font_large.render(system_status, True, display_color)
        screen.blit(status_render, (WIDTH // 2 - status_render.get_width() // 2, HEIGHT // 4))

        # Bottom panel metrics
        y_offset = HEIGHT // 2 + 30
        ml_text = font_medium.render("MACHINE LEARNING: ESP32 SERIAL INPUT", True, GREEN)
        screen.blit(ml_text, (30, y_offset))

        pace_text = font_small.render(f"Established Pace: {current_pace:.2f}s", True, WHITE)
        screen.blit(pace_text, (30, y_offset + 50))

        jitter_text = font_small.render(f"Relative Variance: {rhythm_jitter:.2f}", True, CYAN)
        screen.blit(jitter_text, (400, y_offset + 50))

        trigger_text = font_small.render(f"Arrhythmia Trigger: > {irregularity_trigger:.2f}", True, WHITE)
        screen.blit(trigger_text, (400, y_offset + 80))

        graph_label = font_small.render("Live Voltage (White) vs. Raw Spike (Cyan)", True, GRAY)
        screen.blit(graph_label, (30, y_offset + 120))

        # --- DRAW REAL-TIME GRAPH ---
        graph_rect = pygame.Rect(30, y_offset + 140, WIDTH - 60, 100)
        pygame.draw.rect(screen, (30, 30, 30), graph_rect)
        pygame.draw.rect(screen, (80, 80, 80), graph_rect, 1)
        pygame.draw.line(screen, (60, 60, 60),
                         (30, graph_rect.bottom),
                         (WIDTH - 30, graph_rect.bottom), 2)

        point_spacing = graph_rect.width / max_points
        raw_points = []
        spike_points = []

        for i in range(max_points):
            x = graph_rect.left + (i * point_spacing)
            raw_y = graph_rect.bottom - (raw_history[i] / 4095.0) * graph_rect.height
            spike_y = graph_rect.bottom - (spike_history[i] / 1270.0) * graph_rect.height
            raw_points.append((x, raw_y))
            spike_points.append((x, spike_y))

        if len(raw_points) > 1:
            pygame.draw.lines(screen, WHITE, False, raw_points, 1)
        if len(spike_points) > 1:
            pygame.draw.lines(screen, CYAN, False, spike_points, 2)

        # Update display
        pygame.display.flip()
        clock.tick(60)

    # Cleanup on exit
    ser.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()