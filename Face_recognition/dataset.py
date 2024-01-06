from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class TripletSiameseDataset(Dataset):

    def __init__(self, directory_path, transform=None):
        self.images, self.labels = self.load_dataset(directory_path)
        self.transform = transform
        self.triplets = self.generate_triplets()
        

    def load_dataset(self, directory_path):
        images = []
        labels = []
        current_label = 0

        for person in os.listdir(directory_path):
            person_path = os.path.join(directory_path, person)

            if os.path.isdir(person_path):
                for file in os.listdir(person_path):
                    file_path = os.path.join(person_path, file)

                    img = cv2.imread(file_path)
                    images.append(img)
                    labels.append(current_label)

            current_label +=1

        return images, labels
    

    def generate_triplets(self):
        triplets = []

        for i in range(len(self.images)):
            anchor = self.images[i]
            label = self.labels[i]
            label_to_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}

            positive_index = np.random.choice(label_to_indices[label])
            positive_image = self.images[positive_index]

            negative_label = np.random.choice([element for element in np.unique(self.labels) if element != label])
            negative_index = np.random.choice(label_to_indices[negative_label])
            negative_image = self.images[negative_index]

            triplets.append((anchor, positive_image, negative_image))

        return triplets
    
    
    def __len__(self):
        return len(self.triplets)
    
    
    def __getitem__(self, index):
        anchor, positive, negative = self.triplets[index]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    