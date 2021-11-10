import cv2
from pathlib import Path
import sys

"""
Break up an mp4 video into jpg images of each of its frames.
usage: python3 split_video.py <mp4 file path> <folder path where jpgs will be stored>
"""

if len(sys.argv) != 3:
    print("usage:  python3 split_video.py <mp4 file path> <folder path where jpgs will be stored>")

vid_path = sys.argv[1]
out_dir_path = sys.argv[2]

vid = cv2.VideoCapture(vid_path)
num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(vid.get(cv2.CAP_PROP_FPS) + .5)

out_dir = Path(out_dir_path)
i = 0
frames_to_save = []
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == False:
        break
    
    # only get one frame per second 
    if i % fps == 0:
        frames_to_save.append(frame)

    if i % 1000 == 0:
        for j in range(0, len(frames_to_save)):
            cv2.imwrite(str(out_dir_path / Path(f'frame{i-1000+j}.jpg')),frames_to_save[j])
        print(f'Frame {i}/{num_frames} ({(float(i)/float(num_frames))*100.0}%)')
        frames_to_save = []
    i += 1

for j in range(0, len(frames_to_save)):
    cv2.imwrite(str(out_dir_path / Path(f'frame{i-1000+j}.jpg')),frames_to_save[j])
print(f'Frame {i}/{num_frames} ({(float(i)/float(num_frames))*100.0}%)')
frames_to_save = []

vid.release()
cv2.destroyAllWindows()