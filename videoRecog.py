import cv2
from deepface import DeepFace

video_matches_frames = []

def find_match(frame, ref_img, frame_number):
    try:
        if DeepFace.verify(frame, ref_img.copy(), enforce_detection=False)['verified']:
            video_matches_frames.append(frame_number)
    except ValueError:
        pass


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

    for frame_nr in video_matches_frames:
        seconds = frame_nr // 30
        minutes = seconds // 60 
        seconds = seconds - 60 * minutes 
        print(f"{minutes} minutes {seconds} seconds")
    cap.release()

videoRecog('/reference.jpg', '/video-test.mp4')
