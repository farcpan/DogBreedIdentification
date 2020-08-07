import cv2
from PIL import Image
from torchvision import datasets, models, transforms


def get_single_image_tensor(file_path, debug=False):
    """
    Open image file by OpenCV, convert to PIL Image and convert to PyTorch tensor.

    Parameters
    --------------
    file_path : str
        The file path of image to be converted to PyTorch tensor.
    """
    # OpenCV: [Height, Width, Channel], BGR
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PIL Image (pixel value ranging [0, 1])
    pil_image = Image.fromarray(img)

    noramlized_transform = transforms.Compose([transforms.ToTensor()])
    normalized_image = noramlized_transform(pil_image)

    # [Channel, Height, Width] ---> [BatchSize=1, Channel, Height, Width]
    result_img_tensor = normalized_image.view(
        1, normalized_image.shape[0], normalized_image.shape[1], normalized_image.shape[2])
    return result_img_tensor