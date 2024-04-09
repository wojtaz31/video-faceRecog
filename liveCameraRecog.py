import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)

match = False

ref_img = cv2.imread('ref.jpg')

def startLiveCapture(freq = 30):
    counter = 0
    while True:
        ret, frame = cap.read()
        if counter % freq == 0:
            if not ret:
                continue

            if DeepFace.verify(frame, ref_img.copy(), enforce_detection=False)['verified']:
                cv2.putText(frame, 'MATCH DETECTED', (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            else:
                cv2.putText(frame, 'NO MATCH', (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

        else:
            counter += 1

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

startLiveCapture(120)