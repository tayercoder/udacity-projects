import torch
import numpy as np
from torchvision import models
from PIL import Image
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="path to image to be classified")
    parser.add_argument("model_check_point", help="checkpoint path to load")
    parser.add_argument("--top_k",default=5, type=int, help="number of top K most likely classes")
    parser.add_argument("--category_names", default="cat_to_name.json", help="mapping of categories to names")
    parser.add_argument("--gpu", default="gpu", help="device type, cpu or gpu")
    
    args = vars(parser.parse_args())
    img_path_input = args["image_path"]
    checkpoint_path_input = args["model_check_point"]
    topk_input = args["top_k"]
    categoryname_input = args["category_names"]
    
    #check if category file name exists
    if not os.path.isfile(categoryname_input):
        print(f"Category name file doesn't exist, please provide correct file name.")
        sys.exit()
    
    # reference: https://docs.python.org/2/library/os.path.html
    if os.path.isfile(checkpoint_path_input) and os.path.isfile(img_path_input):
        #select device: cpu or gpu
        device = torch.device("cuda" if torch.cuda.is_available() and args["gpu"] == "gpu" else "cpu")
        
        print(f"Analyzing the image provided...")
        
        # process image so that forward method can handle
        tensor_image = process_image(img_path_input)
        
        #load the model from saved dir
        checkpoint = torch.load(checkpoint_path_input)
        arch = checkpoint["arch"]
        model_loaded = None
        if arch == 'vgg13':
            model_loaded = models.vgg13(pretrained=True)
        elif arch == 'vgg16':
            model_loaded = models.vgg16(pretrained=True)
        else:
            print(f"Currently this app supports only vgg13 or vgg16 models from torchvision, please train your network with one of those network then try predict again.")
            sys.exit()

        for param in model_loaded.parameters():
            param.requires_grad = False
        model_loaded.classifier = checkpoint["classifier"]
        model_loaded.load_state_dict(checkpoint["state_dict"])
        model_loaded.class_to_idx = checkpoint["class_idx_mapping"]
        #End load the model from saved dir
        
        class_idx_mapping = model_loaded.class_to_idx.items()
        class_mapping = {v:k for k, v in class_idx_mapping}
        
        # predict the image using trained network
        probs, classes = predict_image(tensor_image, model_loaded, device, class_mapping, topk_input)
        
        # mapping category id with it's name. 
        #this code below about loading json is provided in the project 1 and I used it as is
        
        with open(categoryname_input, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[c] for c in classes]
        
        print(f"*************** Your image most probably looks like ********************\n")
        for class_name, prob in zip(class_names, probs):
            print(f"Class Name: {class_name}, Predicted Probability: {100 * prob:.2f}%")
        print(f"*************** Thank you for using this AI service! *******************")
        
    else:
        print(f"Please provide correct path for image and checkpoint!")
        sys.exit()


def predict_image(image_tensor, model,device, class_index_mapping, topk=5):
    model.to(device)
    model.eval
        
    with torch.no_grad():
        logps = model.forward(image_tensor.to(device))
    ps = torch.exp(logps)
    probs_top, labels_top = ps.topk(topk)
    probs_top, labels_top = probs_top.to(device), labels_top.to(device)
    
    probs_top = probs_top.cpu().detach().numpy().tolist()[0]
    labels_top = labels_top.cpu().detach().numpy().tolist()[0]
    classelabels = [class_index_mapping[label_idx] for label_idx in labels_top]
    
    return probs_top, classelabels

def process_image(imagepath):
    image = Image.open(imagepath)
    width, height = image.size
    image.thumbnail((256, 10000) if width < height else (10000, 256))
    
    # calculate the cordinates to do center crop
    width1, height1 = image.size
    left = (width1 - 224)/2
    bottom = (height1 - 224)/2
    right = left + 224
    top = bottom + 224
    
    image = image.crop((left, bottom, right, top))
    
    # make image values between 0 and 1
    numpy_image = np.array(image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_image = (numpy_image - mean) / std
    numpy_image = numpy_image.transpose((2,0,1))
    numpy_image = numpy_image[np.newaxis,:]
    
    processed_image = torch.from_numpy(numpy_image)
    processed_image = processed_image.float()
    return processed_image

if __name__ == '__main__': main()