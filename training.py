import cv2 
import numpy as np
import os
#import torch
#from facenet_pytorch import InceptionResnetV1
#from sklearn.svm import SVC
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_path = 'dataset'

def normalize_image(img, color=False, size=True, value=False, order=False):
    """ normalizes the images to size 160x160 with values between -1 and 1
        
        gets the images ready for FaceNet that expects 160x160 size normalized to [-1, 1]"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size:
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)

    if color:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    
    if value:
        img = img.astype(np.float32) / 127.5 - 1.0
    if order:
        img = np.transpose(img, (2, 0, 1))

    return img

def get_labels_and_faces(path, is_dict=False):
    people = os.listdir(path)
    numpy_images = []
    processed_images = []
    if is_dict:
        dataset = {}
    else:
        dataset = []

    for person in people:
        for path, _, images in os.walk(os.path.join(path, person)):
            numpy_images = [cv2.imread(os.path.join(path, img)) for img in images]        
            
            for i in range(len(numpy_images)):
                processed_image = normalize_image(numpy_images[i])
            
                if is_dict:
                    dataset[f'{person}'].append(processed_image)
                else:
                    dataset.append((person, processed_image))

                print(processed_image.shape)
            #print(path)
            #print(images)
    #print(dataset)
    return dataset

def train_recognition_model(data_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = InceptionResnetV1(pretrained= 'vggface2').eval().to(device)

    embeddings = []
    labels = []

    for label, faces in data_dict.items():
        for face in faces:
            if face.shape != (3, 160, 160):
                raise ValueError(f"Image for {label} has wrong shape: {face.shape}. Expected (3, 160, 160)")
            face_tensor = torch.tensor(face).float().to(device).unsqueeze(0)
            embedding = embedder(face_tensor).detach().cpu().numpy()[0]

            embeddings.append(embedding)
            labels.append(label)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # encode string labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # train SVM classifier on the embeddings
    classifier = make_pipeline(StandardScaler(), 
                               SVC(kernel='linear', probability=True))
    classifier.fit(embeddings, encoded_labels)

    return {"classifier" : classifier,
            "label_encoder" : label_encoder,
            "embedder" : embedder}

def train_with_cv2(data_list): 
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    names = [name[0] for name in data_list]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(names)

    faces = [face[1] for face in data_list]
    print(f'names type: {type(names)}, names: {names}, labels: {labels}')
    print(f'faces type: {type(faces)}')

    print('beginning training')
    recognizer.train(faces, np.array(labels))
    print('training done')

    save_path = os.path.join(os.getcwd(), 'trainer')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'training path created: {save_path}')
    
    print(f'[INFO] {len(np.unique(names))} faces trained')
    recognizer.write(os.path.join(save_path, 'trainer.yml'))


if __name__ == '__main__':
    data = get_labels_and_faces(data_path)
    print(len(data))
    train_with_cv2(data)

    print('Exiting')
    

    #recognizer = train_recognition_model(data)

