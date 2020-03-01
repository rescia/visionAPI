#!/usr/bin/env python
# coding: utf-8

# # 画像をAPIに送るテスト

# In[5]:


# 環境変数にセットする
get_ipython().system('export GOOGLE_APPLICATION_CREDENTIALS="./Vision API Project-8a70d6847a65.json"')


# In[1]:


import io
import os
from pathlib import Path
import sys, cv2
from matplotlib import pyplot as plt


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

def convert_absolute(val, orig_size):
    return int(val * orig_size)

# Instantiates a client
client = vision.ImageAnnotatorClient()


# In[2]:


files = []
for seq in range(0, 8):
    for seq_r in range(0, 10):
        st = '../cuttingPicture/dist/cuttedPicture_' + str(seq) + '_' + str(seq_r) + '.jpg'
        files.append(st)
files


# In[3]:


for f_k, f_v in enumerate(files):    
    # The name of the image file to annotate
    target_file_name = f_v
    file_name = os.path.join(
        Path().resolve(),
        target_file_name)
    print(file_name)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    objects = client.object_localization(
        image=image).localized_object_annotations
    # print('Number of objects found: {}' .format(len(objects)))

    # print(objects)
    img = cv2.imread(os.path.join(Path().resolve(),target_file_name), cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape[:3]
    for o_k, obj in enumerate(objects):
        print(obj.name)
        points = []
        for k, vertice in enumerate(obj.bounding_poly.normalized_vertices):
    #     for vertice in obj.bounding_poly.normalized_vertices:
            points.append({
                "x":convert_absolute(vertice.x,width),
                "y":convert_absolute(vertice.y,height)
            })
        cv2.line(img, (points[0]["x"], points[0]["y"]), (points[1]["x"], points[1]["y"]), (255, 0, 0), thickness=10)
        cv2.line(img, (points[1]["x"], points[1]["y"]), (points[2]["x"], points[2]["y"]), (255, 0, 0), thickness=10)
        cv2.line(img, (points[2]["x"], points[2]["y"]), (points[3]["x"], points[3]["y"]), (255, 0, 0), thickness=10)
        cv2.line(img, (points[3]["x"], points[3]["y"]), (points[0]["x"], points[0]["y"]), (255, 0, 0), thickness=10)
        # cv2.line(img, (83, 516), (27, 516), (255, 0, 0), thickness=10)
#         ax = fig.add_subplot(f_k+1, 10, o_k+1)
#         ax.imshow(img)
#         plt.title(obj.name)

    row, col = target_file_name[-7:-6], target_file_name[-5:-4]
    filename = "dist/detected_" +str(row) + "_" + str(col) + ".jpg"
    cv2.imwrite(filename, img)


# In[4]:


# 検証用セクション
for obj in objects:
    print('\n(confidence: {})' .format(obj.name, obj.score))
    print('Normalized bounding polygon vertices: ')
    for vertex in obj.bounding_poly.normalized_vertices:
        print(' - (x=>{},y=>{}\n)' .format(vertex.x, vertex.y))
# print('response:')
# print(response)

# print('Labels:')
# for label in labels:
#     print(label.description)


# In[23]:





# In[33]:


#     ax.title(obj.name)


# In[ ]:




