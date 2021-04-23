import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cv2 as cv2 
import sys
# parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
# parser.add_argument("source", help="Path to the source image")
# parser.add_argument("reference", help="Path to the reference image")
# parser.add_argument("output", help="Path to the output image")
# parser.add_argument('--radius', dest='radius', default=2, help='Radius of the filter kernels (default: 2)')
# parser.add_argument('--sigma-spatial', dest='sigma_spatial', default=2.5, help='Sigma of the spatial weights (default: 2.5)')
# parser.add_argument('--sigma-range', dest='sigma_range', help='Sigma of the range weights (default: standard deviation of the reference image)')
# args = parser.parse_args()

# img_source = 'depth0592.png'
# img_ref = 'color0592.png'
# output = 'output.png'
img_source = 'depth_transformed.png'
img_ref = "color0177.png"
output = 'output_aligned.png'
args_radius = 2
args_sigma_spatial = 1.5
args_sigma_range = None

source_image = Image.open(img_source)

reference_image = Image.open(img_ref)
reference = np.array(reference_image)

source_image_upsampled = source_image.resize(reference_image.size, Image.BILINEAR)
# plt.imshow(source_image_upsampled)
# plt.show()
source_upsampled = np.array(source_image_upsampled)
source_upsampled = np.array([source_upsampled, source_upsampled, source_upsampled]).transpose((1, 2, 0))
print(source_upsampled.shape)
scale = source_image.width / reference_image.width
radius = int(args_radius)
diameter = 2 * radius + 1
step = int(np.ceil(1 / scale))
padding = radius * step
sigma_spatial = float(args_sigma_spatial)
sigma_range = float(args_sigma_range) if args_sigma_range else np.std(reference)
print("?", reference.shape)
reference = np.pad(reference, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)
source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)

# Spatial Gaussian function.
x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
kernel_spatial = np.exp(-1.0 * (x**2 + y**2) /  (2 * sigma_spatial**2))
kernel_spatial = np.repeat(kernel_spatial, 3).reshape(-1, 3)

# Lookup table for range kernel.
lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2))

def process_row(y):
   result = np.zeros((reference_image.width, 3))
   y += padding
   for x in range(padding, reference.shape[1] - padding):
      I_p = reference[y, x]
      patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)
      patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)
      patch_source_upsampled_mask = patch_source_upsampled != 0
      # print("!", patch_source_upsampled)
      kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]
      weight = kernel_range * kernel_spatial
      # print("?", weight[patch_source_upsampled_mask])
      # if np.any(patch_source_upsampled_mask):
      #    print("yup!")
      #    print(">", patch_source_upsampled)
      #    print(">>", patch_source_upsampled_mask)
      #    print(">>>", patch_source_upsampled[patch_source_upsampled_mask])
      # print(patch_source_upsampled.shape, weight.shape, patch_source_upsampled[patch_source_upsampled_mask].shape, patch_source_upsampled_mask.shape)
      k_p = weight[patch_source_upsampled_mask].sum(axis=0)
      #======
      if patch_source_upsampled[patch_source_upsampled_mask].shape[0] != 0:
         # print(patch_source_upsampled[patch_source_upsampled_mask])   
         min_depth = np.amin(patch_source_upsampled[patch_source_upsampled_mask])
         max_depth = np.amax(patch_source_upsampled[patch_source_upsampled_mask])
         depth_delta = max_depth - min_depth

         skip_interpolation_ratio = 0.04693441759

         skip_interpolation_threshold = skip_interpolation_ratio * min_depth
         if depth_delta > skip_interpolation_threshold:
            # print("!", x, y)
            result[x - padding] = source_upsampled[y, x]
         else:
      #======
            if k_p == 0:
               k_p = 1
            # print(weight)
            # exit()
            result[x - padding] = np.round(np.sum(weight[patch_source_upsampled_mask] * patch_source_upsampled[patch_source_upsampled_mask], axis=0) / k_p)
      else:
         if k_p == 0:
            k_p = 1
            # print(weight)
            # exit()
         result[x - padding] = np.round(np.sum(weight[patch_source_upsampled_mask] * patch_source_upsampled[patch_source_upsampled_mask], axis=0) / k_p)
   exit()
   return result

executor = ProcessPoolExecutor(max_workers=8)
result = executor.map(process_row, range(reference_image.height), chunksize=sys.maxsize)
executor.shutdown(True)
# Image.fromarray(np.array(list(result)).astype(np.uint8)).save(output)
result = np.array(list(result))
result = result / np.amax(result) * np.amax(source_image)
# print(result.shape, np.amax(result))
# print(result[0, 0], type(result[0, 0, 0]))
result = np.uint16(result)[:, :, 0]
# print(result.shape, type(result[0, 0]))

# plt.figure(figsize=(8,16))
# plt.imshow(source_image)
# plt.show()

# plt.figure(figsize=(8,16))
# plt.imshow(result)
# plt.show()

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(source_image)
plt.title('Reprojected depth map')
# plt.colorbar()
plt.subplot(122)
plt.imshow(result)
print(type(result[0, 0]))

plt.title(f'Joint bilateral filtering (sigma_spatial={args_sigma_spatial}, sigma_range={sigma_range})')
# plt.colorbar()

plt.show()

# plt.imshow(source_image - result)
# plt.show()

cv2.imwrite(output, result)
