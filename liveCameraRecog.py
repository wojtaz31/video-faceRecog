import cv2
import threading
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)

match = False

def find_match(frame, ref_img):
    global match
    try:
        if DeepFace.verify(frame, ref_img.copy(), enforce_detection=False)['verified']:
            match = True
        else:
            match = False
    except ValueError:
        pass

def startLiveCapture(reference_path ,freq = 30):
    counter = 0
    ref_img = cv2.imread(reference_path)
    while True:
        ret, frame = cap.read()
        if counter % freq == 0:
            if not ret:
                continue

            try:
                threading.Thread(target=find_match, args=(frame.copy(), ref_img)).start()
            except ValueError:
                pass

            if match:
                label = 'MATCH DETECTED'
                color = (0, 255, 0)
            else:
                label = 'NO MATCH'
                color = (255, 0, 0)
        counter += 1
        cv2.putText(frame, label, (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, color)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

startLiveCapture('ref.jpg', 120)