from picamera2 import Picamera2
import sys 
import tty
import termios
import select
import time
import argparse
import os
import cv2

dataset_dir = "dataset"
face_detection_model = "models/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(face_detection_model)
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')

names = os.listdir(dataset_dir)
#print(names)

def get_key():
    """ get keypress from terminal """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        # check if key pressed
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
            return key
        return None 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    print("initializing camera")
    cam = Picamera2()
    preview_config = cam.create_preview_configuration(main={"size": (640, 480)})
    cam.configure(preview_config)
    cam.start(show_preview=True)

    print("press the space bar to capture an image, q to exit")
    image = None

    try:
        while(True):
            key = get_key()
            if key == ' ':
                try:
                    still_config = cam.create_still_configuration(main={"size": (640, 480), "format": 'RGB888'})
                    image = cam.switch_mode_and_capture_array(still_config, "main")
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
                    for (x,y,w,h) in faces:
                        print("face found")
                        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
                        label, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                        if confidence < 100:
                            name = names[label]
                        else:
                            name = 'unknown'
                            
                        conf = f" {round(100 - confidence)}%"
                        
                        print('Hello ' + name + conf)
                        cv2.imshow('image', image)

                except Exception as e:
                    print(f"Exception: {e}")

            elif key == 'q':
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("Terminating...")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera Stopped")
