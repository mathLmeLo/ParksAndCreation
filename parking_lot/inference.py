import torch
import numpy as np
# from PIL import Image

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
#img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
img = 'images/parking_lot_2.png'


# Inference
results = model(img)

def convert_format(input_array):    
    # Extrair os valores de xmin, ymin, xmax e ymax
    output_array = []

    for element in input_array:        
      xmin, ymin, xmax, ymax = element
      # Criar o numpy array de saída no formato desejado
      output_array.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    return np.array(output_array)

# Results
#results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
inference = results.pandas().xyxy[0]
vehicles = inference[inference["name"].isin(["car", "bus", "truck"])]
print(vehicles.loc[:, ["xmin", "ymin", "xmax", 'ymax']].to_numpy().shape)
print(convert_format(vehicles.loc[:, ["xmin", "ymin", "xmax", 'ymax']].to_numpy()).shape)