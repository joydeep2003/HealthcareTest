# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from schemas import ImageUploadResponse


# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         with open(file.filename, "wb") as f:
#             f.write(contents)
#         return JSONResponse(content={"filename": file.filename, "message": "File uploaded successfully"}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"message": str(e)}, status_code=500)

# ==================================================================================================================================================

# WORKING CODE -----------------------------------------
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from schemas import ImageUploadResponse, ErrorResponse
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# from io import BytesIO
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException

# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Load the pre-trained model
# MODEL_PATH = 'pneumoniaAndCovid.h5'
# # model = load_model(MODEL_PATH)
# # model = load_model('/Backend/pneumoniaAndCovid.h5')
# model = tf.keras.models.load_model(MODEL_PATH)


# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         with open(file.filename, "wb") as f:
#             f.write(contents)
#         response_model = ImageUploadResponse(filename=file.filename, message="File uploaded successfully")
#         return response_model.dict()
#     except Exception as e:
#         error_response_model = ErrorResponse(message=str(e))
#         return JSONResponse(content=error_response_model.dict(), status_code=500)
    



# def load_image_into_numpy_array(data):
#     """ Converts the uploaded image file into a numpy array """
#     image = Image.open(BytesIO(data))
#     image = image.resize((224, 224))  # Make sure to resize to the same size as training
#     image_array = np.array(image)
#     image_array = image_array / 255.0  # Normalize the image as we did during training
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     """Receives an uploaded image file and predicts using the pre-trained model."""
#     try:
#         # Validate file extension
#         if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
#             raise HTTPException(status_code=400, detail="Invalid file format.")

#         # Read image file as a numpy array
#         image_data = await file.read()
#         image = load_image_into_numpy_array(image_data)

#         # Make prediction
#         predictions = model.predict(image)
#         predicted_class = np.argmax(predictions, axis=1)[0]

#         # For binary classification, you might return the class directly
#         class_name = 'malignant' if predicted_class == 0 else 'benign'

#         return JSONResponse(status_code=200, content={"class": class_name, "confidence": float(np.max(predictions))})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

# =====================================================================================================================================================


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from schemas import ImageUploadResponse, ErrorResponse
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from io import BytesIO

# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the pre-trained model
# MODEL_PATH = 'pneumoniaAndCovid.h5'
# model = tf.keras.models.load_model(MODEL_PATH)

# def load_image_into_numpy_array(data):
#     """ Converts the uploaded image file into a numpy array """
#     image = Image.open(BytesIO(data))
#     image = image.resize((224, 224))  # Make sure to resize to the same size as training
#     image_array = np.array(image)
#     image_array = image_array / 255.0  # Normalize the image as we did during training
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     """Receives an image file and uploads it."""
#     try:
#         contents = await file.read()
#         with open(file.filename, "wb") as f:
#             f.write(contents)
#         response_model = ImageUploadResponse(filename=file.filename, message="File uploaded successfully")
#         return response_model.dict()
#     except Exception as e:
#         error_response_model = ErrorResponse(message=str(e))
#         return JSONResponse(content=error_response_model.dict(), status_code=500)

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     """Receives an uploaded image file and predicts using the pre-trained model."""
#     try:
#         # Validate file extension
#         if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
#             raise HTTPException(status_code=400, detail="Invalid file format.")

#         # Read image file as a numpy array
#         image_data = await file.read()
#         image = load_image_into_numpy_array(image_data)

#         # Make prediction
#         predictions = model.predict(image)
#         predicted_class = np.argmax(predictions, axis=1)[0]

#         # For binary classification, you might return the class directly
#         class_name = 'malignant' if predicted_class == 0 else 'benign'

#         return JSONResponse(status_code=200, content={"class": class_name, "confidence": float(np.max(predictions))})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

# [][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][]
from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from fastapi.responses import JSONResponse
from schemas import ImageUploadResponse, ErrorResponse, ImageUpload
import tensorflow as tf
import numpy as np
import os
from PIL import Image
# import PIL
from io import BytesIO
import torch
import utils
import exp
# import torch
import torchvision
from captum.attr import * #captun is the library in torch comprises of several explaination module. In our case we're using 'Occulsion'
from torchvision import transforms
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
# MODEL_PATH = 'pneumoniaAndCovid.h5'
# model = tf.keras.models.load_model(MODEL_PATH)

# model = torch.load('model.pth', map_location=torch.device('cpu'))
# model.eval()
# +++++++++++++EXTRAAAAAAAAAAAAAAA++++++++++++++
# def load_image_into_numpy_array(data):
#     """
#     Convert an uploaded image file in byte format into a preprocessed numpy array.
#     """
#     image = Image.open(BytesIO(data))
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     image = np.array(image)
#     image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.LANCZOS3)
#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image.astype(np.float32)
# +++++++++++++++++++++++++++++++++++++++++++++++++++


# image_path = "path/to/image.jpg"
# image = Image.open(image_path)


# ./././...//././././././././/./././././././././././././././././././././.
# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(BytesIO(contents)).convert('RGB')

#     class_name, confidence = utils.predict(image)
#     # exp.explanation(image)

#     return {"class": class_name, "confidence": confidence}

# /'/'/'''''''''''/'/'/'/'/'/'/'/'/''''''''/'/'/'/'/'/'/'/'/'/''''
# @app.post("/explanation/")
# async def explanation_file(file: UploadFile = File(...)):
#     return {"explanation":exp.explanation(image)}

# from threading import Lock

# stored_image_lock = Lock()
# stored_image = None

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     with stored_image_lock:
#         contents = await file.read()
#         image = Image.open(BytesIO(contents)).convert('RGB')
#         stored_image = image

#     class_name, confidence = utils.predict(image)
#     return {"class": class_name, "confidence": confidence}

# @app.post("/explanation/")
# async def explanation_file():
#     with stored_image_lock:
#         if stored_image is None:
#             raise HTTPException(status_code=404, detail="No image has been uploaded yet.")
#         explanation = exp.explanation(stored_image)

#     return {"explanation": explanation}

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO


app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = Form(None),email:str = Form(None)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    user = email.split('@')[0]
    print(user)

    class_name, confidence = utils.predict(image,user)
    # explanation = exp.explanation(image)
    return {"class": class_name, "confidence": confidence}

@app.post("/explanation/")
async def explanation_file(email: str = Form(...)):
    user = email.split('@')[0]
    image_path = user + ".png"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="No image has been uploaded yet.")

    # Load and save the image
    with open(image_path, "rb") as img_file:
        image = Image.open(img_file)
        stored_image = image.copy()

    # Generate explanation using the stored image
    explanation = exp.explanation(stored_image)

    return {"explanation": explanation}







# ========================================================================================================================

# async def upload_image(file: UploadFile = File(...)):
#     """Receives an image file and uploads it."""
#     try:
#         contents = await file.read()
#         with open(file.filename, "wb") as f:
#             f.write(contents)
#         response_model = ImageUploadResponse(filename=file.filename, message="File uploaded successfully")
#         return response_model.dict()
#     except Exception as e:
#         error_response_model = ErrorResponse(message=str(e))
#         return JSONResponse(content=error_response_model.dict(), status_code=500)


# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     """
#     Receives an uploaded image file and predicts using the pre-trained model.
#     """
#     try:
#         if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             raise HTTPException(status_code=400, detail="Invalid file format.")
#         image_data = await file.read()
#         image = load_image_into_numpy_array(image_data)
#         predictions = model.predict(image)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         confidence = float(np.max(predictions))
#         class_name = 'malignant' if predicted_class == 0 else 'benign'
#         return JSONResponse(status_code=200, content={"class": class_name, "confidence": confidence})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
