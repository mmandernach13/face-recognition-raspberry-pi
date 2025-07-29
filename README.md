# Facial Recognition using Raspberry Pi
This project is a Raspberry Pi based program that utilizes the AI Pi Camera with OpenCV to build a dataset, train the model, and recognize faces. It is a proof of concept for facial recognition that could be expanded to a face ID deadbolt or personalized messages on a robot for example. 

## Prerequisites
- Raspberry Pi 4 with Raspbian OS
- AI Pi Camera module (works with USB cam with some modification)
- `dataset` folder for face images

## Installation
1. Clone the repository:
```bash 
    git clone https://github.com/mmandernach13/face-recognition-raspberry-pi.git
    cd face-recogniton-raspberry-pi
```
2. (Optional) make virtual environment:
```bash
    python -m venv venv
    source venv/bin/activate
```
3. Install requirements:
```bash
    pip install -r requirements.txt
```
If you want to use the `train_recognition_model()`, you will also need to install torch, facenet-pytorch, and scikit-learn by uncommenting them in the `requirements.txt`.

## Usage
1. Run the `headshot_cap.py` script to build the dataset of faces with `python headshot_cap.py <name>`
2. Position the camera in front of the person's face, press the spacebar, and get at least 10 (more is better) face detections with various positions.
3. Run the `training.py` script with `python training.py`. This could be changed to use the FaceNet model, but it would require recognition changes also. 
4. Run the `recognition.py` with `python recognition.py`
5. Press the space bar to take a picture, and the identity and confidence will be printed out. 

## License 
This project is licensed under the MIT License.
