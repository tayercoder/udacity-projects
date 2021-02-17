import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", default="flowers", help="Dataset directory")
    
    parser.add_argument("--arch", default="vgg16", help="Model from torchvision.models")
    parser.add_argument("--hidden_units", default=1024, type=int, help="number of units the hidden layer should have")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate for optimizer")
    parser.add_argument("--epochs", default=3, type=int, help="number of epochs")
    parser.add_argument("--gpu", default="gpu", help="device type, cpu or gpu")
    parser.add_argument("--save_dir", default="models", help="directory for model checkpoint")
    
    args = vars(parser.parse_args())
    
    #reading all input args into local var
    data_dir_input = args["data_dir"]
    arch_input = args["arch"]
    hidden_input = args["hidden_units"]
    learningrate_input = args["learning_rate"]
    epochs_input = args["epochs"]
    savedir_input = args["save_dir"]
    
    if not os.path.exists(savedir_input):
        os.mkdir(savedir_input)
    
    train_dir = data_dir_input + '/train'
    valid_dir = data_dir_input + '/valid'
    
    # Define transforms for the training data and validation data
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    traindataset_class_to_index = train_dataset.class_to_idx
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    # END Define transforms for the training data and validation data
    
    #select device: cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args["gpu"] == "gpu" else "cpu")
    
    # prepare the model
    model = None
    if arch_input == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch_input == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print(f"Please enter vgg13 or vgg16 for model architecture --arch")
        sys.exit()
    
    print(f"Training your Neural Network please be patient...")
    
    # Freeze parameters, this code is same as what I learned from UDacity
    for param in model.parameters():
        param.requires_grad = False
    
    # to make sure that second hidden layer's output number is less than it's input number 
    found_hidden_num = hidden_input < 4096
    # if use's input qualifies then use it as is otherwise calculate the correct number
    hidden_num = hidden_input if found_hidden_num else 1024
    while not found_hidden_num:
        hidden_num = int(hidden_input / 2)
        found_hidden_num = hidden_num < 4096
    # End to make sure that second hidden layer's output number is less than it's input number         

    # this code is same as what I learned from UDacity about Sequentioal
    custom_classifier = nn.Sequential(nn.Linear(25088, 4096),
                                      nn.ReLU(),
                                      nn.Linear(4096, hidden_num),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(hidden_num, 102),
                                      nn.LogSoftmax(dim=1))
    model.classifier = custom_classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate_input)
    model.to(device)
    
    # start training, this code is similar to what I learned from UDacity 
    print_every = 20
    steps = 0
    running_loss = 0
    for epoch in range(epochs_input):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch: {epoch+1}/{epochs_input}.. "
                      f"Training Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation Accuracy: {validation_accuracy/len(validloader):.3f}")
                running_loss =0
                model.train()
    # END training
    
    # Save the checkpoint
    model.class_idx_mapping = traindataset_class_to_index
    checkpoint_save_path = savedir_input + '/checkpoint.pth'
    checkpoint = {'epoch':epochs_input,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_idx_mapping': model.class_idx_mapping,
                  'arch':arch_input}
    torch.save(checkpoint, checkpoint_save_path)
    # End Save the checkpoint
    print(f"Your model {arch_input} has been trained and saved successfully to this location: {checkpoint_save_path}!")

    
if __name__ == '__main__':
        main()