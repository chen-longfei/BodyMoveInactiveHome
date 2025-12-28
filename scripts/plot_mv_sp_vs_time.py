import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os

def plot_mv_sp_vs_time(json_path):
    # Load the data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # First, read person_f and t_stamp
    person_f = []
    x = []
    for entry in data:
        if 't_stamp' in entry and 'person_f' in entry:
            x.append(entry['t_stamp'])
            person_f.append(entry['person_f'])
    x = np.array(x)
    person_f = np.array(person_f)

    # Smooth person_f independently (window 11, majority rule)
    person_f_window = 11
    pad = person_f_window // 2
    padded_person_f = np.pad(person_f, (pad, pad), mode='edge')
    smoothed_person_f = np.zeros_like(person_f, dtype=float)
    for i in range(len(person_f)):
        window_vals = padded_person_f[i:i+person_f_window]
        smoothed_person_f[i] = np.mean(window_vals)

    # Now, read y (mv_sp) and set y to -1 wherever smoothed_person_f < 0.5
    # Also build clustering_y for thresholding
    window = 5
    y = []
    clustering_y = []
    for i, entry in enumerate(data):
        if 'mv_sp' in entry:
            if smoothed_person_f[i] < 0.5:
                y.append(-1)
            else:
                y.append(entry['mv_sp'])
                clustering_y.append(entry['mv_sp'])
        else:
            y.append(-1)
    y = np.array(y)
    clustering_y = np.array(clustering_y)

    # Smooth y (only where smoothed_person_f >= 0.5)
    smoothed_y = y.copy()
    valid_indices = smoothed_person_f >= 0.5
    if np.sum(valid_indices) > 0:
        valid_y = y[valid_indices]
        smoothed_valid_y = np.convolve(valid_y, np.ones(window)/window, mode='same')
        smoothed_y[valid_indices] = smoothed_valid_y

    # Learn a threshold using clustering_y only (exclude y=-1)
    if len(clustering_y) > 0:
        clustering_y_reshape = clustering_y.reshape(-1, 1)
        kmeans = KMeans(n_clusters=10, random_state=0).fit(clustering_y_reshape)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        threshold = centers[1]  # boundary between lowest and next cluster
    else:
        threshold = 0.1  # fallback

    # Find low movement periods (at least 1 second continuous below threshold with person present)
    low_move_periods = []
    current_start = None
    for i in range(len(x)):
        if valid_indices[i] and smoothed_y[i] < threshold:
            if current_start is None:
                current_start = x[i]
            # else: already in a period
        else:
            if current_start is not None:
                # Check if the period is at least 1 second
                if x[i-1] - current_start >= 1.0:
                    low_move_periods.append((current_start, x[i-1]))
                current_start = None
    # Handle if the last period goes to the end
    if current_start is not None and x[-1] - current_start >= 1.0:
        low_move_periods.append((current_start, x[-1]))

    # Print the low movement periods
    txt_lines = []
    txt_lines.append('Low movement periods (>=1s):')
    print('Low movement periods (>=1s):')
    for start, end in low_move_periods:
        line = f"  {start:.2f} s  -->  {end:.2f} s  (duration: {end-start:.2f} s)"
        print(line)
        txt_lines.append(line)
    # Save to txt file
    txt_path = os.path.splitext(json_path)[0] + '_low_movement.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')
    print(f"Low movement periods saved to {txt_path}")

    # Plot
    plt.figure(figsize=(10, 5))
    # Plot valid (person present, after smoothing) as filled dots (no line)
    plt.plot(x[valid_indices], smoothed_y[valid_indices], marker='o', linestyle='None', label='smoothed mv_sp (person present)')
    # Plot no-person as empty dots (where smoothed_person_f < 0.5)
    no_person_indices = smoothed_person_f < 0.5
    plt.plot(x[no_person_indices], np.zeros(np.sum(no_person_indices)), marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='gray', label='no person')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('mv_sp')
    plt.title('Smoothed mv_sp vs Time (seconds) with Movement Threshold (10 clusters, exclude no person)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save the plot to the same directory as json_path, with .png extension
    save_path = os.path.splitext(json_path)[0] + '_low_movement_log.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_mv_sp_vs_time(r'homeTest/20250711/homeTest9.56_20250711_data.json') 