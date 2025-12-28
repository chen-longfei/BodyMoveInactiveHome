# Body Inactivity and Speed Estimation

This project provides a system for estimating human inactivity and body speed. It is designed for body monitoring applications, utilizing computer vision techniques to track movement and analyze activity levels over time.

## Description

The system processes video input (from files or webcam) to:
- Detect human poses using a lightweight OpenPose model with MobileNet.
- Separate foreground from background using both pixel-based and pose-based methods.
- Estimate body speed and detect periods of inactivity.
- Log activity data to JSON and text files.
- Visualize the results with overlayed pose skeletons and bounding boxes.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chen-longfei/BodyMoveInactiveHome.git
    cd BodyMoveInactiveHome
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

    *Note: You may need to install PyTorch manually depending on your CUDA version. See [PyTorch.org](https://pytorch.org/).*

3.  **Download Pre-trained Models:**
    The system relies on a pre-trained checkpoint (e.g., `checkpoint_iter_370000.pth`). Ensure this file is present in the root directory or `third_party/lightweight_pose/` directory as expected by the scripts.
    [Download Link (from original repo)](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)

## Usage

Run the main processing script:

```bash
python process_video.py
```

**How it works:**
1.  Running the script will open a **file dialog window**.
2.  Select the **folder** that contains your `.mp4` video files.
3.  The program will automatically find all videos in that folder.
4.  It will process them one by one, showing a progress bar (e.g., `Processing video: 45%`) in the terminal.
5.  Once finished, the results will be saved in a `results/` folder.

## Output

For each processed video, the system generates:
*   `*_processedVid.mp4`: The output video with valid pose overlays and bounding boxes.
*   `*_data.json`: Contains extracted features and timestamps.
*   `*_no_movement_log.txt`: Logs periods of detected inactivity.
*   `*_low_movement_log.png`: A plot of movement speed vs time (if plotting is enabled).

## Credits and Acknowledgements

This project builds upon the excellent work of **Lightweight OpenPose**:
*  [https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch)
    *   Licensed under the Apache License 2.0. See `third_party/lightweight_pose/` for licence details.

## Citation

If you use this repository, please cite the following paper:

*   **OPPH: A Vision-Based Operator for Measuring Body Movements for Personal Healthcare**
    *   *Link*: [https://dl.acm.org/doi/10.1007/978-3-031-92591-7_13](https://dl.acm.org/doi/10.1007/978-3-031-92591-7_13)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






