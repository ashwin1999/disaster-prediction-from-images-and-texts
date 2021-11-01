# Disaster Prediction API

# Description:

This API is built with **FastAPI** and uses state-of-the-art **TensorFlow pretrained models** to detect the occurrence of disaster from both **texts** and **images**. The notebooks containing the code for training these models can be found in the _notebooks folder_.

## Deep Learning Models Used

| Model Name      | Epochs | Accuracy | F1 Score |
| --------------- | ------ | -------- | -------- |
| BERT            | 9      | 76%      | 75%      |
| Efficientnet B3 | 3      | 95%      | NA       |
| ResNet-50       | 3      | 94%      | NA       |

# Routes:

## /one_text/

Method: POST

### **Required:**

text=String

### **Success Response:**

**Payload:** { text : "the car is arriving this evening!" }

**Code:** 200

**Content:** { "prediction" : 0.45, "pred_class" : "Normal" }

**Payload:** { text : "the storm destroyed the buildings that were made of woods." }

**Code:** 200

**Content:** { "prediction" : 0.74, "pred_class" : "Disaster" }

## /multi_text/

Method: POST

### **Required:**

text=[String]

### **Success Response:**

**Payload:** { texts : [ "the storm destroyed the buildings that were made of woods.", "the car is arriving this evening!" ] }

**Code:** 200

**Content:** { "prediction" : [ 0.74, 0.45 ], "pred_class" : [ "Disaster", "Normal" ] }

## /image/

Method: POST

### **Required:**

file = image file

### **Success Response:**

**payload:**

![https://www.who.int/images/default-source/health-and-climate-change/fire-fighters-at-forest-fire-c-quarrie-photography.tmb-479v.jpg?sfvrsn=8b60f828_1%20479w](https://www.who.int/images/default-source/health-and-climate-change/fire-fighters-at-forest-fire-c-quarrie-photography.tmb-479v.jpg?sfvrsn=8b60f828_1%20479w)

**Code:** 200

**Content:** { "prediction" : "Wild Fire" }
