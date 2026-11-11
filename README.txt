RAUSHAN KUMAR - IMAGE AND VIDEO PROCESSING ASSIGNMENT
Subject: 21CSE480T

This package contains:
- Raushan_Kumar_IVP_Final_Code.py: implementations for all 10 problems
- RoadTraffic_Synthetic_Demo.mp4: generated synthetic demonstration input (not real traffic footage)
- output_images/: generated image-processing outputs
- extracted_frames/: frames extracted every 20 frames
- processed_videos/: sampled, filtered and compressed video outputs
- metrics/: JSON and CSV results

Install:
    pip install opencv-python numpy

For H.264 compression and bitrate ladder:
    Install FFmpeg and ensure ffmpeg is available on PATH.

Run using the included synthetic demo:
    python Raushan_Kumar_IVP_Final_Code.py

Run using your own input video:
    python Raushan_Kumar_IVP_Final_Code.py path/to/your_video.mp4

IMPORTANT:
The included input is synthetic and is provided to demonstrate the workflow.
If your faculty expects the same RoadTraffic.mp4 used in your report, replace the
synthetic input with that original file and rerun the program. The metrics in this
package correspond only to the included synthetic demonstration video.
