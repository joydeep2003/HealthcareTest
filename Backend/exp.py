import torch
import torchvision
from captum.attr import * #captun is the library in torch comprises of several explaination module. In our case we're using 'Occulsion'
from torchvision import transforms
import matplotlib.pyplot as plt



model_path = "model.pth" #load the model from whereever the fuck you've saved that
model = torch.load(model_path, map_location='cpu') #CPU mai le model ko

def explanation(image):
    # image_path = "pneu.jpg"
    # image = Image.open(image_path)
    print("here 0")
    occlusion = Occlusion(model) #our explaination module is ready!

    # any network only takes tensors converted from other types of data. So 'image' is the image to test. Upload and make 'image_tensor' from 'image'.
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    image_tensor = preprocess(image).unsqueeze(0)

    print("here after model")
    #this is where the magic happens!
    attributions = occlusion.attribute(image_tensor, sliding_window_shapes=(3, 30, 30), strides=(3, 5, 5))
    importance = torch.sum(attributions.abs(), dim=1, keepdim=True) #we collapse along the dim 1, thanks to the additive structures of tensors lol!
    importance_np = importance[0, 0].detach().numpy()
    image_np = image_tensor[0, 0].detach().numpy()
    importance_np = (importance_np - importance_np.min()) / (importance_np.max() - importance_np.min()) # We min-max everything from [0,1] to match the scale

    # #this is where the magic happens!
    # attributions = occlusion.attribute(image_tensor, sliding_window_shapes=(3, 30, 30), strides=(3, 5, 5))
    # importance = torch.sum(attributions.abs(), dim=1, keepdim=True) #we collapse along the dim 1, thanks to the additive structures of tensors lol!

    importance_np = (importance_np - importance_np.min()) / (importance_np.max() - importance_np.min()) # We min-max everything from [0,1] to match the scale
    print("Here 1")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title('Xray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np, cmap='gray')
    plt.imshow(importance_np, cmap='jet', alpha=0.5)
    plt.title('Severity Map')
    # Adding colorbar
    # cbar = grid.cbar_axes[0].colorbar(img, ticks=cbar_ticks)
    # cbar.ax.set_xticklabels(cbar_labels)
    # cbar.ax.tick_params(labelsize=8)
    # cbar = plt.colorbar()
    # cbar.set_ticks(cbar_ticks)

    # cbar.set_ticklabels(cbar_labels)
    plt.axis('off')

    # plt.show()  *************************************************
    print("Here 2")
    plt.savefig('C:\\Users\\91983\\OneDrive\\Desktop\\Joydeep\\HealthcareTest\\Frontend\\public\\final_image.png')
    plt.close()
    print("Here 3")
    # return 'final_image.png'
    return True

    # add these and remove plt.show()


# def explanation



