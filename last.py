#!/usr/bin/env python3
import glob
import threading
import time
import collections
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import serial
import requests

# --- CONFIG -----------------------------------------------------
BAUD_RATE           = 115200
BUFFER_FRAMES       = 128
SAMPLE_RATE         = 22050
GRID_SIZE_CM        = 60.0
SPEED_OF_SOUND_CM_S = 34300.0

EXPECTED_IDS = {"MIC1", "MIC2", "MIC3"}  # Only 3 microphones

MIC_POS = {
    'MIC1': np.array([0.0, 0.0]),
    'MIC2': np.array([20.0, 0.0]),
    'MIC3': np.array([40.0, 0.0]),
}

BYTES_PER_BLOCK = BUFFER_FRAMES * 4
ENERGY_THRESHOLD = 0.00003
SENSITIVITY = 1.0
DOWNSAMPLE_FACTOR = 2
BUFFER_HISTORY = 4
CLAP_COOLDOWN = 0.10

current_energy_threshold = ENERGY_THRESHOLD
current_sensitivity = SENSITIVITY
lock = threading.Lock()
audio_buffers = {}
last_detection_time = 0

def find_usb_ports():
    return sorted(glob.glob("/dev/ttyUSB*"))

def perform_handshake(timeout=10):
    pending = set(EXPECTED_IDS)
    ports = find_usb_ports()
    device_map = {}
    start_time = time.time()
    while pending and (time.time() - start_time < timeout):
        for dev in ports:
            if not pending:
                break
            try:
                ser = serial.Serial(dev, BAUD_RATE, timeout=0.5)
                time.sleep(0.1)
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line in pending:
                        ser.write(b"ACK\n")
                        ser.flush()
                        device_map[line] = dev
                        pending.remove(line)
                ser.close()
            except Exception:
                pass
        time.sleep(0.1)
    return device_map

def reader_thread(mic_id, dev_path):
    while True:
        try:
            ser = serial.Serial(dev_path, BAUD_RATE, timeout=2)
            ser.reset_input_buffer()
            while True:
                if ser.in_waiting < 4:
                    time.sleep(0.001)
                    continue
                hdr = ser.read(4)
                if len(hdr) < 4:
                    continue
                timestamp = int.from_bytes(hdr, 'little')
                data = ser.read(BYTES_PER_BLOCK)
                if len(data) < BYTES_PER_BLOCK:
                    continue
                samples = np.frombuffer(data, dtype=np.int32).astype(np.float32)
                samples = samples / (2**31)
                if DOWNSAMPLE_FACTOR > 1:
                    samples = signal.decimate(samples, DOWNSAMPLE_FACTOR, zero_phase=True)
                samples = np.clip(samples, -0.5, 0.5)
                with lock:
                    if mic_id not in audio_buffers:
                        audio_buffers[mic_id] = collections.deque(maxlen=BUFFER_HISTORY)
                    audio_buffers[mic_id].append((timestamp, samples.astype(np.float32)))
        except Exception:
            time.sleep(1)

def get_averaged_samples():
    with lock:
        if len(audio_buffers) < 3:
            return None
        averaged_samples = {}
        for mic_id in EXPECTED_IDS:
            if mic_id not in audio_buffers or len(audio_buffers[mic_id]) == 0:
                return None
            recent_samples = list(audio_buffers[mic_id])[-1:]
            if len(recent_samples) == 0:
                return None
            avg_samples = recent_samples[0][1]
            averaged_samples[mic_id] = avg_samples
    return averaged_samples

def get_lane_from_energy_simple(energies, lane_centers, mic_xs):
    sorted_idx = np.argsort(energies)[::-1]
    max_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    max_val = energies[max_idx]
    second_val = energies[second_idx]
    # If two highest energies are close (within 15%), highlight lane between them
    if abs(max_val - second_val) / max_val < 0.30:
        mic_pair = sorted([mic_xs[max_idx], mic_xs[second_idx]])
        mid = (mic_pair[0] + mic_pair[1]) / 2
        lane_idx = np.argmin([abs(center - mid) for center in lane_centers])
        return lane_idx
    else:
        # Otherwise, highlight lane closest to max-energy mic
        mic_pos = mic_xs[max_idx]
        lane_idx = np.argmin([abs(center - mic_pos) for center in lane_centers])
        return lane_idx

def get_lane_from_energy_7lanes(energies, lane_centers, mic_xs):
    # Lane 0: MIC1, Lane 1: between MIC1-MIC2, Lane 2: MIC2, Lane 3: between MIC2-MIC3, ...
    sorted_idx = np.argsort(energies)[::-1]
    max_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    max_val = energies[max_idx]
    second_val = energies[second_idx]

    # If two highest energies are close (within 15%), and are neighbors, pick the lane between them
    if abs(max_val - second_val) / max_val < 0.30 and abs(max_idx - second_idx) == 1:
        # Lane between two mics: always odd lane indices
        lane_idx = min(max_idx, second_idx) * 2 + 1
        return lane_idx
    else:
        # Otherwise, highlight lane at the mic (even indices)
        lane_idx = max_idx * 2
        return lane_idx

def main():
    global current_energy_threshold, current_sensitivity, last_detection_time

    device_map = perform_handshake()
    if not device_map:
        print("[ERROR] No devices found, exiting")
        return

    threads = []
    for mic_id in EXPECTED_IDS:
        if mic_id in device_map:
            thread = threading.Thread(target=reader_thread, args=(mic_id, device_map[mic_id]), daemon=True)
            thread.start()
            threads.append(thread)
        else:
            print(f"[ERROR] No device found for {mic_id}")
            return

    time.sleep(2)

    # --- 5 lanes: mic1, between1-2, mic2, between2-3, mic3 ---
    NUM_LANES = 5
    mic_xs = [MIC_POS['MIC1'][0], MIC_POS['MIC2'][0], MIC_POS['MIC3'][0]]
    lane_centers = [
        mic_xs[0],  # Lane 0: MIC1
        (mic_xs[0] + mic_xs[1]) / 2,  # Lane 1: between MIC1-MIC2
        mic_xs[1],  # Lane 2: MIC2
        (mic_xs[1] + mic_xs[2]) / 2,  # Lane 3: between MIC2-MIC3
        mic_xs[2],  # Lane 4: MIC3
    ]
    lane_memory = np.zeros(NUM_LANES)
    Z = np.zeros((1, NUM_LANES))
    last_pos = np.array([GRID_SIZE_CM / 2, 0.0])

    # For display, set lane boundaries halfway between centers
    lane_edges = [lane_centers[0] - (lane_centers[1] - lane_centers[0]) / 2]
    for i in range(1, len(lane_centers)):
        lane_edges.append((lane_centers[i-1] + lane_centers[i]) / 2)
    lane_edges.append(lane_centers[-1] + (lane_centers[-1] - lane_centers[-2]) / 2)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [2, 1]})

    hm = ax1.imshow(Z, extent=(lane_edges[0], lane_edges[-1], 0, 1),
                    origin='lower', cmap='plasma', alpha=0.8, vmin=0, vmax=1, aspect='auto')
    marker = ax1.scatter([last_pos[0]], [0.5], c='yellow', s=400, marker='*', edgecolor='red', linewidth=3)
    ax1.set_title("Vehicle Lane Detection (1D Mic Array)", fontsize=14, weight='bold')
    ax1.set_xlabel("X (cm)")
    ax1.set_yticks([])
    ax1.set_xticks(lane_centers)
    ax1.grid(True, alpha=0.3, axis='x')

    for mic_id, pos in MIC_POS.items():
        ax1.plot(pos[0], 0.5, 'wo', markersize=12, markeredgecolor='black', linewidth=2)
        ax1.text(pos[0], 0.3, mic_id, ha='center', fontsize=10, color='black', weight='bold')

    # Toggle bar for energy level
    ax_toggle = plt.axes([0.8, 0.92, 0.15, 0.05])
    check = CheckButtons(ax_toggle, ['Show Energy'], [False])
    show_energy = [False]
    def toggle_energy(label):
        show_energy[0] = not show_energy[0]
    check.on_clicked(toggle_energy)

    # Sliders for sensitivity and energy threshold
    ax_slider_sens = plt.axes([0.15, 0.01, 0.3, 0.03])
    slider_sens = Slider(ax_slider_sens, 'Sensitivity', 0.1, 5.0, valinit=current_sensitivity, valfmt='%.2f')
    ax_slider_energy = plt.axes([0.55, 0.01, 0.3, 0.03])
    slider_energy = Slider(ax_slider_energy, 'Energy Thresh', 0.00001, 0.001, valinit=current_energy_threshold, valfmt='%.5f')
    def update_sensitivity(val):
        global current_sensitivity
        current_sensitivity = slider_sens.val
    slider_sens.on_changed(update_sensitivity)
    def update_energy(val):
        global current_energy_threshold
        current_energy_threshold = slider_energy.val
    slider_energy.on_changed(update_energy)

    # Energy bar
    energy_bar = ax1.barh([0.9], [0], height=0.1, color='green', alpha=0.7, label='Energy')
    energy_text = ax1.text((lane_edges[0] + lane_edges[-1])/2, 0.95, '', ha='center', va='bottom', fontsize=12, color='green')

    # Energy levels for all mics
    mic_names = ['MIC1', 'MIC2', 'MIC3']
    mic_bar = ax2.bar(mic_names, [0]*3, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax2.set_title("Microphone Energy Levels", fontsize=12)
    ax2.set_ylabel("Energy")
    ax2.set_ylim(0, 0.01)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.subplots_adjust(right=0.95, top=0.92, bottom=0.08)
    plt.show(block=False)

    print("[INFO] Starting Vehicle Lane Detection system...")

    CAMERA_SERVER_URL = "http://192.168.207.50:5001/camera"
    camera_active = False
    last_lane_idx = None
    camera_start_time = 0
    camera_cooldown_until = 0  # <-- Add this line

    try:
        detection_count = 0
        ENERGY_TRIGGER = 0.006

        while True:
            current_time = time.time()
            samples_dict = get_averaged_samples()
            if samples_dict is None:
                plt.pause(0.005)
                continue

            mic_ids = ['MIC1', 'MIC2', 'MIC3']
            samples = [samples_dict[mic] for mic in mic_ids]
            fs = SAMPLE_RATE // DOWNSAMPLE_FACTOR
            sos = signal.butter(5, [80, 4000], btype='band', fs=fs, output='sos')
            filtered = [signal.sosfilt(sos, s - np.mean(s)) for s in samples]
            energies = [np.sqrt(np.mean(s**2)) for s in filtered]
            max_energy = max(energies)

            for bar, val in zip(mic_bar, energies):
                bar.set_height(val)
                if val * current_sensitivity > current_energy_threshold:
                    bar.set_color('red')
                else:
                    bar.set_color('#4ECDC4')

            if show_energy[0]:
                energy_bar[0].set_width(max_energy * 1000)
                energy_text.set_text(f'Energy: {max_energy:.5f}')
                energy_text.set_color('red' if max_energy * current_sensitivity > current_energy_threshold else 'green')
                energy_bar[0].set_color('red' if max_energy * current_sensitivity > current_energy_threshold else 'green')
                energy_bar[0].set_alpha(0.7)
            else:
                energy_bar[0].set_width(0)
                energy_text.set_text('')

            if max_energy > ENERGY_TRIGGER:
                lane_idx = get_lane_from_energy_5lanes(energies, lane_centers, mic_xs)
                if (current_time > camera_cooldown_until) and (current_time - last_detection_time > CLAP_COOLDOWN):
                    detection_count += 1
                    last_detection_time = current_time
                    Z *= 0.8
                    Z[0, lane_idx] = min(Z[0, lane_idx] + 0.8, 1.0)
                    lane_memory[lane_idx] = 1.0
                    last_pos = np.array([lane_centers[lane_idx], 0.0])
                    print(f"[VEHICLE #{detection_count}] Lane: {lane_idx+1} (X={last_pos[0]:.1f} cm) Energies: {energies}")

                    # --- CAMERA TRIGGER LOGIC (lane number) ---
                    try:
                        resp = requests.post(
                            f"{CAMERA_SERVER_URL}",
                            json={"lane": int(lane_idx+1)}
                        )
                        print(f"[CAMERA] Lane {lane_idx+1} sent, response: {resp.status_code}")
                        camera_cooldown_until = time.time() + 1  # <-- 1 second cooldown
                    except Exception as e:
                        print(f"[CAMERA] Failed to send lane: {e}")
            else:
                # No noise detected, send lane 0 if cooldown expired
                if current_time > camera_cooldown_until:
                    try:
                        resp = requests.post(
                            f"{CAMERA_SERVER_URL}",
                            json={"lane": 0}
                        )
                        print(f"[CAMERA] Lane 0 sent (no noise), response: {resp.status_code}")
                        camera_cooldown_until = time.time() + 1  # <-- 1 second cooldown
                    except Exception as e:
                        print(f"[CAMERA] Failed to send lane 0: {e}")

            lane_memory *= 0.97

            for patch in [p for p in ax1.patches]:
                patch.remove()

            for i in range(NUM_LANES):
                if lane_memory[i] > 0.1:
                    ax1.axvspan(lane_edges[i], lane_edges[i+1],
                                ymin=0, ymax=1, color='lime', alpha=0.15, zorder=0)
            Z *= 0.99
            hm.set_data(Z)
            if Z.max() > 0:
                hm.set_clim(vmin=0, vmax=Z.max())
            marker.set_offsets([[np.clip(last_pos[0], lane_edges[0], lane_edges[-1]), 0.5]])
            fig.canvas.flush_events()
            plt.pause(0.001)
            if plt.waitforbuttonpress(timeout=0.001):
                print("[INFO] Key pressed, exiting main loop.")
                break

    except KeyboardInterrupt:
        print(f"\n[INFO] Shutting down... Total detections: {detection_count}")
    finally:
        for thread in threads:
            thread.join(timeout=1)
        print("[INFO] All threads terminated.")

# --- Lane detection for 5 lanes ---
def get_lane_from_energy_5lanes(energies, lane_centers, mic_xs):
    sorted_idx = np.argsort(energies)[::-1]
    max_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    max_val = energies[max_idx]
    second_val = energies[second_idx]
    # If two highest energies are close (within 30%), highlight lane between them
    if abs(max_val - second_val) / max_val < 0.30:
        mic_pair = sorted([mic_xs[max_idx], mic_xs[second_idx]])
        mid = (mic_pair[0] + mic_pair[1]) / 2
        lane_idx = np.argmin([abs(center - mid) for center in lane_centers])
        return lane_idx
    else:
        # Otherwise, highlight lane closest to max-energy mic
        mic_pos = mic_xs[max_idx]
        lane_idx = np.argmin([abs(center - mic_pos) for center in lane_centers])
        return lane_idx

if __name__ == "__main__":
    main()