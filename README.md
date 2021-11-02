# Disaster Prediction API

# Description:

This API is built with **FastAPI** and uses state-of-the-art models from _TensorFlow Hub_ to detect the occurrence of disaster from both _texts_ and _images_. The notebooks containing the code for training these models can be found in the _notebooks folder_.

## Deep Learning Models Used

| Model Name      | Epochs | Accuracy | F1 Score |
| --------------- | ------ | -------- | -------- |
| BERT            | 9      | 76%      | 75%      |
| Efficientnet B3 | 3      | 95%      | NA       |
| ResNet-50       | 3      | 94%      | NA       |

# Getting Started

Setup a virtual environment and activate it

```
python -m venv your_env
```

Install all the dependencies from the **requirements.txt** file to your environment

```
pip install -r requirements.txt
```

To start your server, run the below command in the terminal

```
uvicorn main:app --reload
```

By default the server will start on port **8000** which can be accessed by opening **http://localhost:8000** on the browser

# Routes

## /one_text/

Method: POST

### **Required**

text=String

### **Success Response**

**Payload:** { text : "the car is arriving this evening!" }

**Code:** 200

**Content:** { "prediction" : 0.45, "pred_class" : "Normal" }

**Payload:** { text : "the storm destroyed the buildings that were made of woods." }

**Code:** 200

**Content:** { "prediction" : 0.74, "pred_class" : "Disaster" }

## /multi_text/

Method: POST

### **Required**

text=[String]

### **Success Response**

**Payload:** { texts : [ "the storm destroyed the buildings that were made of woods.", "the car is arriving this evening!" ] }

**Code:** 200

**Content:** { "prediction" : [ 0.74, 0.45 ], "pred_class" : [ "Disaster", "Normal" ] }

## /image/

Method: POST

### **Required**

file = image file

### **Success Response**

**payload:**

![https://www.who.int/images/default-source/health-and-climate-change/fire-fighters-at-forest-fire-c-quarrie-photography.tmb-479v.jpg?sfvrsn=8b60f828_1%20479w](https://www.who.int/images/default-source/health-and-climate-change/fire-fighters-at-forest-fire-c-quarrie-photography.tmb-479v.jpg?sfvrsn=8b60f828_1%20479w)

**Code:** 200

**Content:** { "prediction" : "Wild Fire" }
