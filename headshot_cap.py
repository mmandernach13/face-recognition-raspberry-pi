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
    parser = argparse.ArgumentParser(description='headshot capturing for face recognition')
    parser.add_argument('name', help='input the name of the person to be added to the recognition model')
    args = parser.parse_args()

    output_dir = os.path.join(dataset_dir, args.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"created output dir: {output_dir}")

    print("initializing camera")
    cam = Picamera2()
    preview_config = cam.create_preview_configuration(main={"size": (640, 480)})
    cam.configure(preview_config)
    cam.start(show_preview=True)

    print("press the space bar to capture an image, q to exit")
    headshot_count = 1
    image = None

    try:
        while(True):
            key = get_key()
            if key == ' ':
                capture = f"headshot{headshot_count:03d}.png"
                save_path = os.path.join(output_dir, capture)
                try:
                    still_config = cam.create_still_configuration(main={"size": (640, 480), "format": 'RGB888'})
                    image = cam.switch_mode_and_capture_array(still_config, "main")
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
                    for (x,y,w,h) in faces:
                        print("face found")
                        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.imwrite(save_path, image[y:y+h, x:x+w])
                        cv2.imshow('image', image)
                        print(f"image saved to {save_path}")
                        headshot_count += 1

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
