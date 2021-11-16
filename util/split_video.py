import cv2
from pathlib import Path
import sys

"""
Break up an mp4 video into jpg images of each of its frames.
usage: python3 split_video.py <mp4 file path> <folder path where jpgs will be stored> <# frames to split into>
"""

if len(sys.argv) != 4:
    print("usage:  python3 split_video.py <mp4 file path> <folder path where jpgs will be stored> <# frames to split into>")

vid_path = sys.argv[1]
out_dir_path = sys.argv[2]
num_frames_to_save = int(sys.argv[3])

vid = cv2.VideoCapture(vid_path)
num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
split_on_frame = int(num_frames / num_frames_to_save)

out_dir = Path(out_dir_path)

# while(vid.isOpened() and i <= 1000):
#     vid_prog = (i*split_on_frame)/num_frames
#     print(vid_prog)
#     vid.set(2, vid_prog) # select next frame
#     ret, frame = vid.read()
#     if ret == False:
#         print("End of Video!")
#         break
    
#     # only get one frame per 1 minute 
#     #if i % (fps*60) == 0:
#         #frames_to_save.append(frame)

#     #if i % (1000) == 0:
#         #for j in range(0, len(frames_to_save)):
#     cv2.imwrite(str(out_dir_path / Path(f'frame{i}.jpg')),frame)
#     #print(f'Frame {i}/{num_frames} ({(float(i)/float(num_frames))*100.0:.0f}%)')
#     print(f"{1/num_frames_to_save:.0f}%")
#         #frames_to_save = []
#     i += 1

# for j in range(0, len(frames_to_save)):
#     c
# print(f'Frame {i}/{num_frames} ({(float(i)/float(num_frames))*100.0:.0f}%)')
# frames_to_save = []

# read entire thing to memory
frames = []
i = 0
percent = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == False:
        break
    # frames.append(frame)
    i += 1
    new_percent = int((float(i) / float(num_frames)) * 100.0)
    if i % split_on_frame == 0:
        cv2.imwrite(str(out_dir_path / Path(f'frame{i}.jpg')),frame)
    # print(f"{i}/{num_frames}")
    if new_percent > percent:
        percent = new_percent
        print(f"Reading frames {percent:.0f}%")

# for i in range(split_on_frame,num_frames,split_on_frame):
#     if i < len(frames):
#         cv2.imwrite(str(out_dir_path / Path(f'frame{i}.jpg')),frames[i])
#     print(f"Writing frames {i/num_frames:.0f}%")
    


vid.release()
cv2.destroyAllWindows()