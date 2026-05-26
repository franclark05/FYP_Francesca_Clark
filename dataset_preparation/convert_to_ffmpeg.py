import os
import subprocess
import cv2

# -------- CONFIG --------
input_dir = r"raw"       #Folder containing .MOV files
output_dir = r"fixed"    #Folder to save converted files
target_fps = 240                              
crf_value = 18                              
# ------------------------


def get_input_fps(video_path):
    """Get the original video FPS using OpenCV (approximate)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def convert_video(input_path, output_path, target_fps, crf):
    """
    Convert video to constant frame rate (CFR) with H.264 codec.
    This ensures OpenCV can seek frames accurately.
    """
    # ffmpeg command
    FFmpeg_PATH = r"C:\PATH_Programs\ffmpeg.exe"
    cmd = [
        FFmpeg_PATH, 
        "-y", 
        "-i", input_path, 
        "-vsync", "cfr", 
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-preset", "veryfast", 
        "-crf", str(crf), 
        "-pix_fmt", "yuv420p", 
        "-an", 
        output_path
    ]

    print(f"\nConverting: {os.path.basename(input_path)}")
    print(f"   Target FPS: {target_fps}, Output: {os.path.basename(output_path)}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Saved: {output_path}")


def batch_convert(input_dir, output_dir, target_fps, crf):
    """Convert all .MOV files in a folder to .mp4 with constant FPS."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mov")]
    if not files:
        print("No .MOV files found in input directory.")
        return

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".mp4")

        #Get original FPS for logging 
        orig_fps = get_input_fps(input_path)
        print(f"\n🎥 {file}: Detected original FPS ≈ {orig_fps:.2f}")

        #Convert the video
        convert_video(input_path, output_path, target_fps, crf)

    print("\nAll conversions complete.")
    print(f"Converted files saved to: {output_dir}")


if __name__ == "__main__":
    batch_convert(input_dir, output_dir, target_fps, crf_value)
