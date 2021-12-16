import torch
from torchvision import datasets, models, transforms
import os
import pandas as pd
import sys
import time
from torchsummary import summary

model_path = sys.argv[1]
model = torch.load(model_path)
experiments_path = "base_truth_data"

def get_train_files_path(experiments_path):
    file_path = os.path.join(experiments_path, 'test.csv')
    train_df = pd.read_csv(file_path, delimiter=',')
    files_path = []
    fonts_class = []
    for row in train_df.iterrows():
        files_path.append(os.path.join(experiments_path, row[1]['class'], row[1]['filename']))
        fonts_class.append(row[1]['class'])
    
    return files_path, fonts_class

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

X_test, y_test = get_train_files_path(experiments_path)

image_dataset = datasets.ImageFolder(os.path.join(experiments_path, "test"), data_transforms["test"])
class_names = image_dataset.classes
dataloader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=16, 
                                             shuffle=True, 
                                             num_workers=4) 
model.to(device)
summary(model, (3,224,224))
model.eval()
total_correct = 0
total_tested = 0

incorrect_images = []
images_shown = 0
incorrect_labels = {x : 0 for x in class_names}
l_count = {x:[0,0] for x in class_names}

total_time = 0

for inputs,labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    since = time.time()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    total_time += time.time() - since
    for i,pred in enumerate(preds):
        if(pred != labels[i] and images_shown < 10):
            images_shown += 1
            incorrect_images.append(inputs[i])
            l_count[class_names[labels[i]]][1] += 1
            #imshow(inputs[i],"Predicted: {} Actual: {}".format(class_names[pred],class_names[labels[i]]))
        else:
            l_count[class_names[labels[i]]][0] += 1
    total_correct += torch.sum(preds == labels.data)
    total_tested += len(inputs.data)

print("Test acc: {} ({}/{})".format(float(total_correct)/float(total_tested),total_correct,total_tested))
for class_name in class_names:
    print("{}/{} correct for {}".format(l_count[class_name][0],l_count[class_name][0]+l_count[class_name][1],class_name))

print("Average inference time per image: {}".format(total_time/total_tested))