import cv2
import numpy as np
import random


class Augmenter:
    """
    A collection of dataset augmentation methods including transformations, 
    blurring, resizing, and brightness adjustments. 

    NOTE: This class is used to transform data necessary for training TechTrack's models.
          Imagine that the output of `self.transform()` is fed directly to train the model.
    
    The following transformations are included:
    - Horizontal flipping: i.e., def horizontal_flip(**kwargs)
    - Gaussian blurring: i.e., def gaussian_blur_image(**kwargs)
    - Resizing: i.e., def resize(**kwargs)
    - Brightness and contrast adjustments: i.e., def change_brightness(**kwargs)
        - HINT: you may use cv2.addWeighted()

    NOTE: These methods uses **kwargs to accept arbitrary keyword arguments,
    but explicit parameter definitions improve clarity and usability.
    - "**kwargs" reference: https://www.geeksforgeeks.org/args-kwargs-python/

    Finally, Provide a demonstration and visualizations of these methods in `notebooks/augmentation.ipynb`.
    You will define your own keywords for "**kwargs".
    """

    
    @staticmethod
    def horizontal_flip(**kwargs):
        """
        Horizontally flip the image.
        
        """
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("horizontal_flip requires kwarg 'image'.")

        p = float(kwargs.get("p", 1.0))
        if random.random() > p:
            return image

        return cv2.flip(image, 1)

    @staticmethod
    def gaussian_blur(**kwargs):
        """
        Apply Gaussian blur to the image.
        
        """
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("gaussian_blur requires kwarg 'image'.")

        p = float(kwargs.get("p", 1.0))
        if random.random() > p:
            return image

        ksize = kwargs.get("ksize", 5)
        sigma = float(kwargs.get("sigma", 0.0))

        # Normalize ksize to (w,h) and ensure odd positive values
        if isinstance(ksize, int):
            k = max(1, int(ksize))
            if k % 2 == 0:
                k += 1
            ksize_tuple = (k, k)
        else:
            kw, kh = int(ksize[0]), int(ksize[1])
            kw = max(1, kw + (1 - kw % 2)) if kw % 2 == 0 else max(1, kw)
            kh = max(1, kh + (1 - kh % 2)) if kh % 2 == 0 else max(1, kh)
            ksize_tuple = (kw, kh)

        return cv2.GaussianBlur(image, ksize_tuple, sigmaX=sigma, sigmaY=sigma)


    @staticmethod
    def resize(**kwargs):
        """
        Resize the image.
        
        """
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("resize requires kwarg 'image'.")

        p = float(kwargs.get("p", 1.0))
        if random.random() > p:
            return image

        interpolation = kwargs.get("interpolation", cv2.INTER_LINEAR)
        size = kwargs.get("size", None)

        if size is not None:
            w, h = int(size[0]), int(size[1])
            if w <= 0 or h <= 0:
                raise ValueError("resize kwarg 'size' must be positive (width, height).")
            return cv2.resize(image, (w, h), interpolation=interpolation)

        fx = float(kwargs.get("fx", 1.0))
        fy = float(kwargs.get("fy", 1.0))
        if fx <= 0 or fy <= 0:
            raise ValueError("resize scale factors fx and fy must be > 0.")

        return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)

    @staticmethod
    def change_brightness(**kwargs):
        """
        Adjust brightness and contrast of the image.
        
        """
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("change_brightness requires kwarg 'image'.")

        p = float(kwargs.get("p", 1.0))
        if random.random() > p:
            return image

        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 0.0))

        return cv2.addWeighted(image, alpha, image, 0.0, beta)

    @staticmethod
    def transform(**kwargs):
        """
        Apply random augmentations from the available methods.
        
        Internal Process:
        1. A list of available augmentation functions is created.
        2. The list is shuffled to introduce randomness.
        3. A random number of augmentations is selected.
        4. The selected augmentations are applied sequentially to the image.
        
        :param image: Input image (numpy array)
        :param kwargs: Additional parameters for transformations (if any)
        :return: Augmented image
        """
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("transform requires kwarg 'image'.")

        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(int(seed))

        methods = [
            Augmenter.horizontal_flip,
            Augmenter.gaussian_blur,
            Augmenter.resize,
            Augmenter.change_brightness,
        ]

        # Shuffle
        random.shuffle(methods)

        # Choose how many
        n = kwargs.get("n", None)
        if n is None:
            min_n = int(kwargs.get("min_n", 1))
            max_n = int(kwargs.get("max_n", len(methods)))
            min_n = max(0, min(min_n, len(methods)))
            max_n = max(min_n, min(max_n, len(methods)))
            n = random.randint(min_n, max_n)
        else:
            n = int(n)
            n = max(0, min(n, len(methods)))

        chosen = methods[:n]

        
        params = kwargs.get("params", {}) or {}

        # 4) Apply sequentially
        out = image
        for fn in chosen:
            name = fn.__name__
            fn_kwargs = dict(params.get(name, {}))
            fn_kwargs["image"] = out
            out = fn(**fn_kwargs)

        return out
        

"""
EXAMPLE RUNNER:

# Create an instance of Augmenter
augmenter = Augmenter()

kwargs = {"image": your_image, # Numpy type
            ... # Add more...
        }

# Apply random transformations
augmented_image = augmenter.transform(**kwargs)

# Display the original and transformed images
cv2.imshow("Original Image", image)
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
