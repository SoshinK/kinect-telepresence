import numpy as np
import cv2 as cv2 
from numba import jit, prange

@jit
def process_image(
    source_upsampled_padded, 
    reference_image_padded, 
    lut_range, 
    padding, 
    kernel_spatial, 
    step,
    skip_interpolation_ratio,
    result):

    for row in range(reference_image_padded.shape[0] - 2 * padding):
        y = row + padding
        for x in range(padding, reference_image_padded.shape[1] - padding):
            I_p = reference_image_padded[y, x]

            k_p = 0
            filtered = 0

            min_depth = 1e5
            max_depth = 0

            for i in range(y - padding, y + padding + 1, step):
                for j in range(x - padding, x + padding + 1, step):
                    if source_upsampled_padded[i, j] != 0:
                        if source_upsampled_padded[i, j] < min_depth:
                            min_depth = source_upsampled_padded[i, j]
                        if source_upsampled_padded[i, j] > max_depth:
                            max_depth = source_upsampled_padded[i, j]

                        kernel_range = lut_range[int(np.abs(reference_image_padded[i, j] - I_p))]
                        k_p += kernel_range * kernel_spatial[(i - y + padding) // step, (j - x + padding) // step]
                        filtered += source_upsampled_padded[i, j] * kernel_range * kernel_spatial[(i - y + padding) // step, (j - x + padding) // step]
            delta_depth = max_depth - min_depth
            interp_thresh = min_depth * skip_interpolation_ratio
            if delta_depth > interp_thresh:
                result[row, x - padding] = source_upsampled_padded[y, x]
            else:
                if k_p == 0:
                    k_p = 1

                result[row, x - padding] = round(filtered / k_p)

class JBU:
    def __init__(self, radius, 
                        sigma_spatial, 
                        sigma_range,
                        scale,
                        skip_interpolation_ratio=0.04693441759): 
        self.radius = int(radius)
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = float(sigma_range)

        self.skip_interpolation_ratio = skip_interpolation_ratio

        self.diameter = 2 * self.radius + 1

        self.scale = scale
        self.step = int(np.ceil(1 / scale))
        self.padding = self.radius * self.step

        # Spatial Gaussian function
        x, y = np.meshgrid(np.arange(self.diameter) - self.radius, np.arange(self.diameter) - self.radius)
        self.kernel_spatial = np.exp(-1.0 * (x ** 2 + y ** 2) /  (2 * self.sigma_spatial ** 2))
        self.kernel_spatial = self.kernel_spatial.reshape((2 * self.padding + 1, 2 * self.padding + 1))

        # Lookup table for range kernel.
        self.lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * self.sigma_range**2))


    def __call__(self, source_image, reference_image):
        self.source_upsampled = cv2.resize(source_image, (reference_image.shape[1], reference_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        
        result = np.zeros((reference_image.shape[0], reference_image.shape[1]))
        

        self.reference_image = np.pad(reference_image, ((self.padding, self.padding), (self.padding, self.padding)), 'symmetric').astype(np.float32)
        self.source_upsampled = np.pad(self.source_upsampled, ((self.padding, self.padding), (self.padding, self.padding)), 'symmetric').astype(np.float32)
        
        process_image(
            self.source_upsampled, 
            self.reference_image, 
            self.lut_range,
            self.padding, 
            self.kernel_spatial.astype(np.float32), 
            self.step,
            self.skip_interpolation_ratio,
            result
            )
        
        result = result / np.amax(result) * np.amax(source_image)
        result = np.uint16(result)
        return result
