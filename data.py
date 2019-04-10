import cv2
import pandas
import numpy as np
import json
import tqdm 
# Loads all useful data from label file
# [Image Name, Lanes, Driverable Area, Non-Driverable Aera, Cars]
def load_data_from_labels(path, to_load):
  count = 0
  with open(path) as json_file:  
    data = json.load(json_file)
    
    formatted_data = []
    
    for entry in tqdm(data):
      
      if count > to_load:
        continue
      
      image_name = entry['name']
      labels = entry['labels']
      
      lanes = []
      drive_area = []
      nondrive_area []
      cars = []

      for label in labels:
        cat = label['category']
        print(cat)
       
        if cat in 'drivable area':
          area_type = label['attributes']['areaType']
          #print(area_type)
          if area_type in 'direct':
            polygon = label['poly2d'][0]
            verts  = polygon['vertices']      
            driveable.append(verts)
          else:
            polygon = label['poly2d'][0]
            verts  = polygon['vertices']      
            alt.append(verts)
        elif cat in 'lane':
          polygon = label['poly2d'][0]
          verts  = polygon['vertices']      
          lanes.append(verts)
      
      formatted_data.append([image_name,lanes, driveable, alt, cars])
      count += 1
     
    print("Loaded " + str(len(formatted_data)) + " entries")
    return formatted_data

def data_to_image(labels):
  return

