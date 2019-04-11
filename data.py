import cv2
import pandas
import numpy as np
import json
import tqdm 
from skimage.draw import line


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
      nondrive_area = []
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

def data_to_image(labels, out_size=(254,126)):

  #Load all sub data from formatted label array
  # [image_name, lane_points, driveable_poly, alt_poly, car_box]
  lanes     = labels[1]
  driveable = labels[2]
  alt       = labels[3]
  cars      = labels[4]

  # ----------------
  #   LANE PARSING
  # ----------------

  # Create blank image to draw on for lanes
  image_lanes = np.zeros([int(720),int(1280),3])

  for cur in lanes:
    y1 = int(cur[0][0] / DOWNSCALE)
    x1 = int(cur[0][1] / DOWNSCALE)
    y2 = int(cur[1][0] / DOWNSCALE)
    x2 = int(cur[1][1] / DOWNSCALE)

    rr, cc = line(x1,y1,x2,y2)
    rr = np.clip(rr, 0, int(720) - 2)
    cc = np.clip(cc, 0, int(1280) -2)

    image_lanes[rr     ,cc, :] = 1.0
    image_lanes[rr     ,cc - 1, :] = 1.0
    image_lanes[rr     ,cc + 1, :] = 1.0
    image_lanes[rr - 1 ,cc , :] = 1.0
    image_lanes[rr + 1 ,cc , :] = 1.0
    image_lanes[rr - 1 ,cc - 1, :] = 1.0
    image_lanes[rr + 1 ,cc + 1, :] = 1.0

  image_lanes = cv2.resize(image_lanes, out_size)
  
  # ----------------
  # DRIVE/ALT PARSING
  # ----------------

  image_area = np.zeros([int(720 ),int(1280),3])
  
  for points in driveable:
    points = np.array([points], dtype=np.int32)
    image = cv2.fillPoly(image_area,points, (0,1.0,0))
    
  for points in alt:
    points = np.array([points], dtype=np.int32)
    image = cv2.fillPoly(image_area,points, (1.0,0,0))

  image_area = cv2.resize(image_area, out_size)

  # ---------------
  #  CAR PARSING
  # ---------------

  image_cars = np.zeros([int(720 ),int(1280),3])
  
  for box in cars:
    box = np.array(box, dtype=np.int32)

    image_cars = cv2.rectangle(image_cars,(box[0][0],box[0][1]),(box[1][0],box[1][1]), (1.0,1.0,1.0), cv2.FILLED)
    
  
  image_cars = cv2.resize(image_cars, out_size)

  return [image_lanes, image_area, image_cars]

def save_data(path, images, index):
  np.save(path + "lane_images-"+str(index)+".npy",images[0])
  np.save(path + "area_images-"+str(index)+".npy",images[1])
  np.save(path + "car_images-"+str(index)+".npy" ,images[2])


def load_all():
  TO_LOAD_TRAIN = 10
  TO_LOAD_VALID = 10

  labels_train = load_data_from_labels('data/bdd100k/labels/bdd100k_labels_images_train.json', TO_LOAD_TRAIN)
  labels_valid = load_data_from_labels('data/bdd100k/labels/bdd100k_labels_images_val.json', TO_LOAD_VALID)

  images_train = []
  images_valid = []

  count = 0
  for image in images_train:
    save_data('data/train/', data_to_image(image),count)
    count += 1
  
  count = 0
  for image in images_valid:
    save_data('data/valid/' + data_to_image(image),count)
    count += 1

 

load_all()
