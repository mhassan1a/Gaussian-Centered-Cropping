import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import crop
from torchvision.transforms import transforms
from mpl_toolkits.axes_grid1 import ImageGrid

class GaussianCrops:
    def __init__(self, crop_percentage=0.4,
                 seed=None, 
                 std_scale=1 ,
                 padding=False, 
                 regularised_crop=False,
                 adaptive_center=False,
                 min_max=(0.25, 0.75)):
        """Initialization method for the cropping class.
            
            Args:
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                std_scale (float, optional): Scaling factor for the standard deviation of the Gaussian distribution. Defaults to 1.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
                adaptive_center (bool, optional): Boolean flag indicating whether the center of the cropping is adaptive. Defaults to False.
                min_max (tuple, optional): Tuple containing the minimum and maximum values for the adaptive center. Defaults to (0.25, 0.75).
            """
        self.crop_percentage = crop_percentage
        self.seed = seed
        self.std_scale = std_scale
        self.padding = padding
        self.regularised_crop = regularised_crop
        self.adaptive_center = adaptive_center
        self.min_max = min_max
        
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
        try:
            assert 0 < self.crop_percentage < 1
        except AssertionError:
            raise ValueError("The crop percentage must be between 0 and 1.")
        try:
            assert self.std_scale > 0
        except AssertionError:
            raise ValueError("The standard deviation scale must be greater than 0.")
        try:
            assert self.min_max[0] < self.min_max[1]
        except AssertionError:
            raise ValueError("The minimum value must be less than the maximum value.")
        try:
            assert 0 <= self.min_max[0] <= 1 and 0 <= self.min_max[1] <= 1
        except AssertionError:
            raise ValueError("The minimum and maximum values must be between 0 and 1.")
        try:
            img.ndim == 3
        except AssertionError:
            raise ValueError("The image must be a 3-dimensional array.")
        
        if img.shape[0] == 3: # C x H x W
            image_height, image_width = img.shape[1], img.shape[2]
            chw = True
        elif img.shape[2] == 3: # H x W x C  
            image_height, image_width = img.shape[0], img.shape[1] 
            chw = False
        else:
            raise ValueError("The image must have 3 channels.")
        
            
        crop_width = int(image_width * np.sqrt(self.crop_percentage))
        crop_height = int(image_height * np.sqrt(self.crop_percentage))
        
        std_x = image_width  * self.std_scale
        std_y = image_height  * self.std_scale
        mean_x, mean_y = [image_width // 2, image_height // 2]
        
        if self.adaptive_center:
            mean_x = np.random.uniform(image_width * self.min_max[0], image_width * self.min_max[1])
            mean_y = np.random.uniform(image_height * self.min_max[0], image_height * self.min_max[1])
        
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = np.random.multivariate_normal([mean_x,mean_y], [[std_x, 0], [0, std_y]], 2)
        
        crops = []
        for center in centers:
            center_x, center_y = int(center[0]), int(center[1])
            left = center_x - crop_width // 2
            top =  center_y - crop_height // 2
            bottom = top + crop_height 
            right = left + crop_width 
            pad_bottom, pad_top, pad_left, pad_right = 0, 0, 0, 0
            
            if self.regularised_crop:
                if top < 0:
                    top = 0
                    bottom = crop_height
                if left < 0:
                    left = 0
                    right = crop_width
                if bottom > image_height:
                    bottom = image_height
                    top = bottom - crop_height
                if right > image_width:
                    right = image_width
                    left = right - crop_width             
                   
       
            if self.padding:
                if left < 0 and right> 0:
                    left = 0
                    pad_left = crop_width - right
                    pad_right = 0
                elif right > image_width and left < image_width:
                    right = image_width
                    pad_left = 0
                    pad_right = right - image_width
                elif left <= 0 and right <=0:
                    left = 0
                    right = 0
                    pad_left = crop_width
                    pad_right = 0 
                elif left >= image_width and right >= image_width:
                    left = 0
                    right = 0
                    pad_left = 0
                    pad_right = crop_width                   
                else:
                    pad_left = 0
                    pad_right = 0
                    
                if top < 0 and bottom > 0:
                    top = 0
                    pad_top = crop_height - bottom
                    pad_bottom = 0
                elif bottom > image_height and top < image_height:
                    bottom = image_height
                    pad_top = 0
                    pad_bottom = bottom - image_height
                elif top <= 0 and bottom <= 0:
                    top = 0
                    bottom = 0
                    pad_top = crop_height
                    pad_bottom = 0
                elif top >= image_height and bottom >= image_height:
                    top = 0
                    bottom = 0
                    pad_top = 0
                    pad_bottom = crop_height
                else:
                    pad_top = 0
                    pad_bottom = 0
                 
            if bottom - top < crop_height:
                pad_top = (crop_height - (bottom - top)) // 2
                pad_bottom = crop_height - (bottom - top) - pad_top   
            if right - left < crop_width:
                pad_left = (crop_width - (right - left)) // 2
                pad_right = crop_width - (right - left) - pad_left
             
            if chw: 
                cropped_image = img[:, top:bottom, left:right]
                cropped_image = np.pad(cropped_image, ((0, 0), (pad_top, pad_bottom),
                    (pad_left, pad_right)), mode='constant')
            else:     
                cropped_image = img[top:bottom, left:right,:]
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                    (pad_left, pad_right), (0, 0)), mode='constant')
             
            crops.append(cropped_image)
        return crops
    
    
def gaussianCrops(crop_percentage=0.4,
                 seed=None, 
                 std_scale=1 ,
                 padding=False, 
                 regularised_crop=False,
                 adaptive_center=False,
                 min_max=(0.25, 0.75)):
        """Initialization method for the cropping class.
            
            Args:
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                std_scale (float, optional): Scaling factor for the standard deviation of the Gaussian distribution. Defaults to 1.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
                adaptive_center (bool, optional): Boolean flag indicating whether the center of the cropping is adaptive. Defaults to False.
                min_max (tuple, optional): Tuple containing the minimum and maximum values for the adaptive center. Defaults to (0.25, 0.75).
            """
        
        def gcc(img):
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
            try:
                assert 0 < crop_percentage < 1
            except AssertionError:
                raise ValueError("The crop percentage must be between 0 and 1.")
            try:
                assert std_scale > 0
            except AssertionError:
                raise ValueError("The standard deviation scale must be greater than 0.")
            try:
                assert min_max[0] < min_max[1]
            except AssertionError:
                raise ValueError("The minimum value must be less than the maximum value.")
            try:
                assert 0 <= min_max[0] <= 1 and 0 <= min_max[1] <= 1
            except AssertionError:
                raise ValueError("The minimum and maximum values must be between 0 and 1.")
            try:
                img.ndim == 3
            except AssertionError:
                raise ValueError("The image must be a 3-dimensional array.")
            
            if img.shape[0] == 3: # C x H x W
                image_height, image_width = img.shape[1], img.shape[2]
                chw = True
            elif img.shape[2] == 3: # H x W x C  
                image_height, image_width = img.shape[0], img.shape[1] 
                chw = False
            else:
                raise ValueError("The image must have 3 channels.")
            
                
            crop_width = int(image_width * np.sqrt(crop_percentage))
            crop_height = int(image_height * np.sqrt(crop_percentage))
            
            std_x = image_width  * std_scale
            std_y = image_height  * std_scale
            mean_x, mean_y = image_width // 2, image_height // 2
            
            if seed is not None:
                    rng = np.random.default_rng(seed)
            rng = np.random.default_rng()
            
            
            if adaptive_center:
                # means = [[mean_x-min_max[0]*image_width, mean_y-min_max[0]*image_height],
                #             [mean_x+min_max[0]*image_width, mean_y-min_max[0]*image_height],
                #             [mean_x-min_max[0]*image_width, mean_y+min_max[0]*image_height],
                #             [mean_x+min_max[0]*image_width, mean_y+min_max[0]*image_height],
                #             [mean_x, mean_y]]
                
                # 
                # mixure = [rng.multivariate_normal(means[i], [[std_x, 0], [0, std_y]], 2) for i in range(5)]
                # centers = mixure[rng.choice(np.arange(5), p=[0.2, 0.2, 0.2, 0.2, 0.2])]
                            
                mean_x = rng.uniform(min_max[0],min_max[1]) * image_width
                mean_y = rng.uniform(min_max[0],min_max[1]) * image_height
                # center_x = std_x * rng.normal(size=(2,1))+ mean_x 
                # center_y = std_y * rng.normal(size=(2,1))+ mean_x 
                # centers = list(zip(center_x, center_y))                
   
            centers = rng.multivariate_normal(np.array([mean_x,mean_y]), [[std_x, 0], [0, std_y]], 2)

        
            boxes = []  
            crops = []
            for center in centers:
                center_x, center_y = int(center[0]), int(center[1])
                left = center_x - crop_width // 2
                top =  center_y - crop_height // 2
                bottom = top + crop_height 
                right = left + crop_width 
                pad_bottom, pad_top, pad_left, pad_right = 0, 0, 0, 0
                
                if regularised_crop:
                    if top < 0:
                        top = 0
                        bottom = crop_height
                    if left < 0:
                        left = 0
                        right = crop_width
                    if bottom > image_height:
                        bottom = image_height
                        top = bottom - crop_height
                    if right > image_width:
                        right = image_width
                        left = right - crop_width             
                    
        
                if padding:
                    if left < 0 and right> 0:
                        left = 0
                        pad_left = crop_width - right
                        pad_right = 0
                    elif right > image_width and left < image_width:
                        right = image_width
                        pad_left = 0
                        pad_right = right - image_width
                    elif left <= 0 and right <=0:
                        left = 0
                        right = 0
                        pad_left = crop_width
                        pad_right = 0 
                    elif left >= image_width and right >= image_width:
                        left = 0
                        right = 0
                        pad_left = 0
                        pad_right = crop_width                   
                    else:
                        pad_left = 0
                        pad_right = 0
                        
                    if top < 0 and bottom > 0:
                        top = 0
                        pad_top = crop_height - bottom
                        pad_bottom = 0
                    elif bottom > image_height and top < image_height:
                        bottom = image_height
                        pad_top = 0
                        pad_bottom = bottom - image_height
                    elif top <= 0 and bottom <= 0:
                        top = 0
                        bottom = 0
                        pad_top = crop_height
                        pad_bottom = 0
                    elif top >= image_height and bottom >= image_height:
                        top = 0
                        bottom = 0
                        pad_top = 0
                        pad_bottom = crop_height
                    else:
                        pad_top = 0
                        pad_bottom = 0
                    
                if bottom - top < crop_height:
                    pad_top = (crop_height - (bottom - top)) // 2
                    pad_bottom = crop_height - (bottom - top) - pad_top   
                if right - left < crop_width:
                    pad_left = (crop_width - (right - left)) // 2
                    pad_right = crop_width - (right - left) - pad_left
                
                if chw: 
                    cropped_image = img[:, top:bottom, left:right]
                    cropped_image = np.pad(cropped_image, ((0, 0), (pad_top, pad_bottom),
                        (pad_left, pad_right)), mode='constant')
                else:     
                    cropped_image = img[top:bottom, left:right,:]
                    cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                        (pad_left, pad_right), (0, 0)), mode='constant')
                
                crops.append(cropped_image)
                boxes.append([top, bottom, left, right])
            return crops, boxes, (centers,(mean_x, mean_y))
        return gcc
        
  
    
if __name__ == "__main__":
    img = Image.open("others/g44yy1dz.bmp")
    img = np.array(img)

    num_samples = 12
    num_views = 2
    padding = True
    regularised_crop = False

    # Create a new figure
    crops1 = []
    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        adaptive_center = True
        std_scale = 1
        crop = GaussianCrops(crop_percentage=crop_percentage,
                            seed=seed,
                            std_scale=std_scale,
                            padding=padding,
                            regularised_crop=regularised_crop,
                            adaptive_center=adaptive_center)
        cropped_images = crop.gcc(img)
        crops1.extend(cropped_images)
    num_images = len(crops1)
    spacing = 0.05
    num_cols = 6
    num_rows = (num_images + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=spacing,  # spacing between images
                     )

    for ax, image in zip(grid, crops1):
        ax.imshow(image)
        ax.axis("off")
    plt.title("Gaussian Cropping with adaptive center")
    plt.legend()
    plt.show()
    plt.savefig(f"gcc_adp_std_{std_scale}_cropsize_{crop_percentage}.jpg")
    crops2 = []
    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        adaptive_center = False
        std_scale = 10
        crop = GaussianCrops(crop_percentage=crop_percentage,
                            seed=seed,
                            std_scale=std_scale,
                            padding=padding,
                            regularised_crop=regularised_crop,
                            adaptive_center=adaptive_center,
                            min_max=(0.4, 0.6))
        cropped_images = crop.gcc(img)
        crops2.extend(cropped_images)
    num_images = len(crops2)
    spacing = 0.05
    num_cols = 6
    num_rows = (num_images + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=spacing,  # spacing between images
                     )

    for ax, image in zip(grid, crops2):
        ax.imshow(image)
        ax.axis("off")
    plt.title("Gaussian Cropping without adaptive center")
    plt.show()
    plt.savefig(f"gcc_not_adp_std_{std_scale}_cropsize_{crop_percentage}.jpg")
    
    
    # Test the cropping class
    img = np.random.rand(3,100,200)
    crop = gaussianCrops(crop_percentage=0.4,
                         seed=42,
                         std_scale=1, 
                         padding=True,
                         regularised_crop=False,
                         adaptive_center=False, 
                         min_max=(0.25, 0.75))
    crops,*_ = crop(img)
    print(crops[0].shape, crops[1].shape)
    