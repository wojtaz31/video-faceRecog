import cv2
from deepface import DeepFace

video_matches_frames = []

def find_match(frame, ref_img, frame_number):
    try:
        if DeepFace.verify(frame, ref_img.copy(), enforce_detection=False)['verified']:
            video_matches_frames.append(frame_number)
    except ValueError:
        pass


def framesToTimestamps(frames, detectIntervals=False):
    timestamps = []
    for frame_nr in frames:
        seconds = frame_nr // 30
        minutes = seconds // 60
        seconds = seconds - 60 * minutes
        if minutes < 10: minutes = '0' + str(minutes)
        if seconds < 10: seconds = '0' + str(seconds)
        timestamps.append(f"{minutes}:{seconds}")
    timestamps = list(dict.fromkeys(timestamps))

    if detectIntervals:
        intervalTimestamps = []
        e = prev = ''
        for i in timestamps:
            if e == '':
                e += i + ' -'
                prev = i
                continue
            if not (int(prev[-2::]) == int(i[-2::]) - 1 or (prev[-2::] == '59' and i[-2::] == '00')):
                e += ' ' + prev
                intervalTimestamps.append(e)
                e = ''
            prev = i
        if e[-1] == '-':
            e += ' ' + timestamps[-1]
            intervalTimestamps.append(e)
        return intervalTimestamps
    return timestamps


def videoRecog(reference_path, videoPath=''):
    frame_number = 0
    ref_img = cv2.imread(reference_path)
    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    while True:
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            break

        if frame_number % 15 == 0:
            find_match(frame.copy(), ref_img, frame_number)
    
    timestamps = framesToTimestamps(video_matches_frames)
    intervalTimestamps = framesToTimestamps(video_matches_frames, True)
    print(timestamps, '\n', intervalTimestamps)
    print(f'Twarz znajduje się w filmie przez {len(timestamps)} sekund')
    cap.release()

videoRecog('/reference.jpg', '/video-test.mp4')