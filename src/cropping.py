import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import crop
from torchvision.transforms import transforms

class GaussianCrops:
    def __init__(self, crop_percentage=0.4, seed=None, std_scale=1 ,
                 padding=False, regularised_crop=False):
        """Initialization method for the cropping class.
            
            Args:
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                std_scale (float, optional): Scaling factor for the standard deviation of the Gaussian distribution. Defaults to 1.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
            """
        self.crop_percentage = crop_percentage
        self.seed = seed
        self.std_scale = std_scale
        self.padding = padding
        self.regularised_crop = regularised_crop
        
    def gcc(self, img):
        """
        Gaussian centered Cropping 
        Crop an image using isotropic Gaussian distributed cropping.
        Parameters:
        - image (np.array): Image to be cropped.
        - crop_percentage (float): Percentage of the original image to be cropped.
        - seed (int): Seed for reproducibility.
        - std_scale (float): Scaling factor for the standard deviation of the Gaussian distribution.
        
        Returns:
        - List of np.arrays: with two views of the cropped image.     
        """
        width, height = img.shape[1], img.shape[0]
        crop_width = int(width * np.sqrt(self.crop_percentage))
        crop_height = int(height * np.sqrt(self.crop_percentage))
        
        std_x = width  * self.std_scale
        std_y = height  * self.std_scale
        
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = np.random.multivariate_normal([width / 2, height / 2], [[std_x, 0], [0, std_y]], 2)
        crops = []
        for center in centers:
            center_x, center_y = center
            left = int(max(0, int(center_x - crop_width / 2)))
            top = int(max(0, int(center_y - crop_height / 2)))
            bottom = top + crop_height if top + crop_height < height else height
            right = left + crop_width if left + crop_width < width else width
            cropped_image = img[top:bottom, left:right]
            
            if self.regularised_crop:
                if bottom - top < crop_height:
                    top = bottom - crop_height
                if right - left < crop_width:
                    left = right - crop_width
                    
            cropped_image = img[top:bottom, left:right]
            
        
            if self.padding:
                cropped_image = img[top:bottom, left:right]
                pad_top = max(0, crop_height - (bottom - top))
                pad_bottom = max(0, crop_height - (bottom - top))
                pad_left = max(0, crop_width - (right - left))
                pad_right = max(0, crop_width - (right - left))
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                    (pad_left, pad_right), (0, 0)), mode='constant')
                
            if cropped_image.shape[0] > crop_height:
                cropped_image = cropped_image[:crop_height]
            if cropped_image.shape[1] > crop_width:
                cropped_image = cropped_image[:, :crop_width]
            
            if cropped_image.shape[0] < crop_height:
                pad_top = (crop_height - cropped_image.shape[0]) // 2
                pad_bottom = crop_height - cropped_image.shape[0] - pad_top
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
            if cropped_image.shape[1] < crop_width:
                pad_left = (crop_width - cropped_image.shape[1]) // 2
                pad_right = crop_width - cropped_image.shape[1] - pad_left
                cropped_image = np.pad(cropped_image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
                
            crops.append(cropped_image) 
                    
        return crops
    
class UniformCrops:
    def __init__(self, min_scale=-1, max_scale =1, crop_percentage=0.4, seed=None,
                padding=False, regularised_crop=False):
        """Initialization method for the cropping class.

            Args:
                min_scale (int, optional): Minimum scale factor for cropping. Defaults to -1.
                max_scale (int, optional): Maximum scale factor for cropping. Defaults to 1.
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
        """
        
        self.crop_percentage = crop_percentage
        self.seed = seed
        self.padding = padding
        self.regularised_crop = regularised_crop
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    
    def ucc(self, img):
        """
        Uniform centered Cropping 
        Crop an image using isotropic uniform distributed cropping.
        Parameters:
        - image (np.array): Image to be cropped.
        - crop_percentage (float): Percentage of the original image to be cropped.
        - seed (int): Seed for reproducibility.
        
        Returns:
        - List of np.arrays: with two views of the cropped image.     
        """
        width, height = img.shape[1], img.shape[0]
        crop_width = int(width * np.sqrt(self.crop_percentage))
        crop_height = int(height * np.sqrt(self.crop_percentage))
        
        min_hight = height//2 - int(width * self.min_scale)
        max_hight=  height//2 + int(width * self.max_scale)
        
        min_width = width//2 - int(height * self.min_scale)
        max_width = width//2 + int(height * self.max_scale)
        
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = np.random.uniform([width / 2, height / 2],
                                    [[min_hight, min_width], [max_hight, max_width]], (2, 2))
        crops = []
        for center in centers:
            center_x, center_y = center
            left = int(max(0, int(center_x - crop_width / 2)))
            top = int(max(0, int(center_y - crop_height / 2)))
            bottom = top + crop_height if top + crop_height < height else height
            right = left + crop_width if left + crop_width < width else width
            cropped_image = img[top:bottom, left:right]
            
            if self.regularised_crop:
                if bottom - top < crop_height:
                    top = bottom - crop_height
                if right - left < crop_width:
                    left = right - crop_width
                    
            cropped_image = img[top:bottom, left:right]
        
            if self.padding:
                cropped_image = img[top:bottom, left:right]
                pad_top = max(0, crop_height - (bottom - top))
                pad_bottom = max(0, crop_height - (bottom - top))
                pad_left = max(0, crop_width - (right - left))
                pad_right = max(0, crop_width - (right - left))
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                    (pad_left, pad_right), (0, 0)), mode='constant')
                
            if cropped_image.shape[0] > crop_height:
                cropped_image = cropped_image[:crop_height]
            if cropped_image.shape[1] > crop_width:
                cropped_image = cropped_image[:, :crop_width]
                
            if cropped_image.shape[0] < crop_height:
                pad_top = (crop_height - cropped_image.shape[0]) // 2
                pad_bottom = crop_height - cropped_image.shape[0] - pad_top
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
                
            if cropped_image.shape[1] < crop_width:
                pad_left = (crop_width - cropped_image.shape[1]) // 2
                pad_right = crop_width - cropped_image.shape[1] - pad_left
                cropped_image = np.pad(cropped_image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
                
            crops.append(cropped_image)
            
        return crops

if __name__ == "__main__":
    img = Image.open("pic1.jpg")
    img = np.array(img)


    # gaussian centered cropping    
    num_samples = 10
    num_views = 2
    images = [] 

    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(num_samples * num_views)))

    # Create a new figure
    plt.figure(figsize=(10, 10))

    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        std_scale = 100
        crop = GaussianCrops(crop_percentage = crop_percentage, seed = seed, 
                            std_scale = std_scale, padding=False,regularised_crop=True)
        cropped_images = crop.gcc(img)#,  regularised_crop=True)
        images.append(cropped_images)
        # Display cropped images
        for j, cropped_image in enumerate(cropped_images):
            # Calculate the subplot index
            index = i * num_views + j + 1
            plt.subplot(grid_size, grid_size, index)
            plt.imshow(cropped_image)
            plt.title("Sample {}, View {}".format(i+1, j+1))

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # uniform centered cropping
    num_samples = 10
    num_views = 2
    images = []

    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(num_samples * num_views)))

    # Create a new figure
    plt.figure(figsize=(10, 10))   

    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        min_scale = -0.5
        max_scale = 0.5
        
        crop = UniformCrops(crop_percentage = crop_percentage, seed = seed, padding=True, 
                            regularised_crop=False, min_scale=min_scale, max_scale=max_scale)
        cropped_images = crop.ucc(img)
        images.append(cropped_images)
        # Display cropped images
        for j, cropped_image in enumerate(cropped_images):
            # Calculate the subplot index
            index = i * num_views + j + 1
            plt.subplot(grid_size, grid_size, index)
            plt.imshow(cropped_image)
            plt.title("Sample {}, View {}".format(i+1, j+1))

    # Adjust layout
    plt.tight_layout()
    plt.show()