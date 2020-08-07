from PIL import Image
from torchvision import models

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch

from model import model
from util import get_tensor
from util import get_device


def load_saved_model(model_path, device, num_classes):
    base_model = models.resnext101_32x8d(pretrained=False)
    loaded_model = model.DogClassificationModel(
        model=base_model, num_classes=num_classes, mean=0.5, std=0.25)
    loaded_model = loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    return loaded_model


def predict(image_path, model, device, debug=False):
    result_image = None
    result_breed_id = 0

    input_tensor = get_tensor.get_single_image_tensor(image_path, debug=False)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, preds = torch.max(output, 1)
        
        for i, value in enumerate(preds):
            result_image = input_tensor[i]
            result_breed_id = value
            break

    df = pd.read_csv("./labels/labels.csv")
    data = df[df["breed_id"] == int(result_breed_id)]
    name = data["breed"].iat[0]

    return name, output


if __name__ == "__main__":
    num_classes = 121
    model_path = "./trained_model"
    image_file_path = "./test/0a0b97441050bba8e733506de4655ea1.jpg"
    
    device = get_device.get_device()
    loaded_model = load_saved_model(model_path, device, num_classes=num_classes)

    start = time.time()

    predicted_name, raw_output = predict(image_file_path, loaded_model, device, debug=False)
    print("Predicted breed: {}, Prediction time: {} [sec]".format(predicted_name, time.time() - start))

    output_values = np.reshape(raw_output.to("cpu").detach().numpy(), (num_classes, ))

    # probability
    prob = np.exp(output_values) / np.sum(np.exp(output_values))
