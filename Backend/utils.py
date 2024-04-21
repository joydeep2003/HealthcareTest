import torch
from torchvision import transforms
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import Image

model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

def predict(image, user):
    # Save the input image
    image.save(f"{user}.png")  # save the image with the username

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output[0], dim=0)

    classes = ["COVID", "PNEUMONIA"]
    class_id = torch.argmax(probabilities).item()
    class_name = classes[class_id]
    confidence = probabilities[class_id].item()
    
    return class_name, confidence

# Example usage:
# from PIL import Image
# img = Image.open('path_to_image.jpg')
# result, confidence = predict(img, 'user')





# ************************************Working code *********************************************************************
# import torch
# from torchvision import transforms
# import numpy as np
# from captum.attr import IntegratedGradients
# import torch.nn.functional as F

# model = torch.load('model.pth', map_location=torch.device('cpu'))
# model.eval()

# def predict(image,user):
 
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = preprocess(image).unsqueeze(0)  #batch dimension

#     # prediction
#     with torch.no_grad():
#         output = model(image)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
   
#     classes = ["COVID","PNEUMONIA"]
#     # prediction
#     class_id = torch.argmax(probabilities).item()
#     class_name = classes[class_id]
#     confidence = probabilities[class_id].item()
#     print("Inside Predict befiore retrninh")
#     return class_name, confidence


# **********************************************************************************************************************************


# def predict(image):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image_tensor = preprocess(image).unsqueeze(0)  #batch dimension
#     classes = ["COVID","PNEUMONIA"]
#     # Prediction
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = F.softmax(output[0], dim=0)

    
#     class_id = torch.argmax(probabilities).item()
#     class_name = classes[class_id]
#     confidence = probabilities[class_id].item()


#     ig = IntegratedGradients(model)
#     attributions, delta = ig.attribute(image_tensor, target=class_id, return_convergence_delta=True)
#     attributions = attributions.sum(dim=1, keepdim=True)

#     # Normalize stuffs
#     attributions = (attributions - torch.min(attributions)) / (torch.max(attributions) - torch.min(attributions))
#     attributions = attributions.detach().cpu().numpy()[0]

#     return class_name, confidence, attributions