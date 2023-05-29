''' Introducing Image Processing and scikit-image '''

'''
Make images come alive with scikit-image
What is image processing?
- It is a method to perform operations on images, in order to:
* Enhance an image
* Extract useful information
* Analyze it and make decisions
- It is a subset of computer vision

Applications of image processing
- Medical image analysis
- Artificial intelligence
- Image restoration and enhancement
- Geospatial computing
- Surveillance 
- Robotic vision
- Automotive safety
- And many more . . . 

Purposes of image processing
- Visualization:
* To observe objects that are not visible
- Image sharpening and restoration
* To create a better image
- Image retrieval
* To seek for the image of interest
- Measurement of pattern
* To measure objects
- Image Recognition
* To distinguish objects in an image

Intro to scikit-image
- It is an image processing library in Python that is easy to use
- It makes use of Machine Learning with built-in functions
- It can perform complex operations on images with just a few functions

What is an image?
- A digital image is an array, or a matrix, of quare pixels (picture elements) arranged in columns and rows: i.e a 2-Dimensional matrix.
- These pixels contains information about color and intensity.

Images in scikit-image()
- There are some testing-purpose images provided by scikit-image, in a module called data.

from skimage import data
rocket_image = data.rocket()

RGB vs Grayscale
- RGB images have 3 color channels, while grayscaled ones have a single channel.

from skimage import color
grayscale = color.rgb2gray(original)
rgb = color.gray2rgb(grayscale)

Visualizing images inthis course
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

from skimage import color
grayscale = color.rgb2gray(original)

show_image(grayscale, 'Grayscale')
'''

# Import the modules from skimage

# Load the rocket image
from skimage import transform
from skimage.restoration import denoise_tv_chambolle, inpaint
from skimage.feature import corner_harris, corner_peaks
from skimage import data, measure
from skimage.transform import rotate, rescale
from skimage import data, exposure
from skimage.color import rgb2gray
from skimage import data, color
rocket = data.rocket()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)

# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')


'''
NumPy for images
- We can practice simple image processing techniques with numpy, such as
* Flipping images
* Extracting and analyzing features

Images as NdArrays
# Loading the image using Matplotlib
madrid_image = plt.imread('/madrid.jpeg')
type(madrid_image) -> in

<class 'numpy.ndarray'> -> out

Colors with NumPy
# Obtaining the red values of the image
red = image[:, :. 0]

# Obtaining the green values of the image
green = image[:, :. 1]

# Obtaining the blue values of the image
blue = image[:, :. 2]

plt.imshow(red, cmap='gray')
plt.title('Red')
plt.axis('off')
plt.show()

Shapes
# Accessing the shape of the image
madrid_image.shape -> in

(426, 640, 3) -> out

Sizes
# Accessing the shape of the image
madrid_image.size -> in

817920 -> out

Flipping images: vertically
# Flip the image in up direction
vertically_flipped = np.flipud(madrid_image)

show_image(vertically_flipped, 'Vertically flipped image') 

Flipping images: horizontally
# Flip the image in left direction
horizontally_flipped = np.fliplr(madrid_image)

show_image(horizontally_flipped, 'Horizontally flipped image') 

What is a histogram?
- The histogram of an image is a graphical representation of the amount of pixels of each intensity value.
* From 0 (pure black) to 255 (pure white)

Color histograms
- In this case, each channel: red, green and blue will have a corresponding histogram.

Applications of histograms
- Analysis
- Thresholding
- To alter brightness and contrast
- To equalize an image

Histograms in Matplotlib
# Red color of the image
red = image[:, :, 0]

# Obtain the red histogram
plt.hist(red.ravel(), bins=256)

- ravel() is used to return a continuous flattened array from the color values of the image

blue = image[:, :, 2]

plt.hist(blue.ravel(), bins=256)
plt.title('Blue Histogram')
plt.show()
'''

# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Flip the image horizontally
seville_horizontal_flip = np.fliplr(seville_vertical_flip)

# Show the resulting image
show_image(seville_horizontal_flip, 'Seville')


# Obtain the red channel
red_channel = image[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)

# Set title and show
plt.title('Red Histogram')
plt.show()


'''
Getting started with thresholding
Thresholding
- It is used to partition the background and foreground of grayscale images.
* by essentially making them black and white.
- We do so by setting each pixel to:
* 255 (white) if pixel > threshold value
* 0 (black) if pixel < threshold value

- It is the simplest method of image segmentation
- It lets us isolate elements and it is used in:
* object detection
* Face detection
* Etc.
- It works best in high-contrast grayscale images.

Original image -> Grayscale image -> Threshold image

Applying Thresholding
# Obtain the optimal threshold value
thresh = 127

# Apply thresholding to the image
binary = image > thresh

# Show the original and thresholded images
show_image(image, 'Original')
show_image(binary, 'Thresholded')

Inverted thresholding
# Obtain the optimal threshold value
thresh = 127

# Apply thresholding to the image
inverted_binary = image <= thresh

# Show the original and thresholded images
show_image(image, 'Original')
show_image(inverted_binary, 'Inverted thresholded')

categories
- Global or histogram based
* It is good for images that have relatively uniform backgrounds
- Local or adaptive:
* It is best for where the background is not easily differentiated, with uneven background illumination

- NB*: Local is slower than global thresholding.

Try more thresholding algorithms
- set Verbose to False so it doesn't print function name for each method.

from skimage.filters import try_all_threshold

# Obtain all the resulting images
fig, ax = try_all_threshold(image, verbose=False)

# Showing resulting plots
show_plot(fig, ax)

Optimal thresh value
- Global
* Uniform background

# Import the otsu threshold function
from skimage.filters import threshold_otsu

# Obtain the optimal threshold value
thresh = threshold_otsu(image)

# Apply thresholding to the image
binary_global = image > thresh

# Show the original and binarized image
show_image(image, 'Original')
show_image(binary_global, 'Global thresholding')

- Local
* Uneven background
- The block_size is also known as the local neighbourhoods 
* it is used to calculate thresholds in small pixel regions, surrounding each pixel we are binarizing.
- An optional offset is a constant subtracted from the mean of blocks to calculate the local threshold value.

# Import the local threshold function
from skimage.filters import threshold_local

# Set the block size to 35
block_size = 35

# Obtain the optimal local thresholding
local_thresh = threshold_local(text_image, block_size, offset=10)

# Apply local thresholding and obtain the binary image
binary_local = text_image > local_thresh

# Show the original and binarized image
show_image(image, 'Original')
show_image(binary_local, 'Local thresholding')
'''

# Import the otsu threshold function

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the image
show_image(binary, 'Binary image')


# Import the otsu threshold function

# Obtain the optimal otsu global thresh value
global_thresh = threshold_otsu(page_image)

# Obtain the binary image by applying global thresholding
binary_global = page_image > global_thresh

# Show the binary image obtained
show_image(binary_global, 'Global thresholding')


# Import the local threshold function

# Set the block size to 35
block_size = 35

# Obtain the optimal local thresholding
local_thresh = threshold_local(page_image, block_size, offset=10)

# Obtain the binary image by applying local thresholding
binary_local = page_image > local_thresh

# Show the binary image
show_image(binary_local, 'Local thresholding')


# Import the try all function

# Import the rgb to gray convertor function

# Turn the fruits_image to grayscale
grayscale = rgb2gray(fruits_image)

# Use the try all method on the resulting grayscale image
fig, ax = try_all_threshold(grayscale, verbose=False)

# Show the resulting plots
plt.show()


# Import threshold and gray convertor functions

# Turn the image grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal thresh
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh

# Show the resulting binary image
show_image(binary_image, 'Binarized image')


''' Filters, Contrast, Transformation and Morphology '''

'''
Jump into filtering
- Filtering is a technique for modifying or enhancing an image.
- Filters can be used to emphasize or remove certain features, like edges.
- Smoothing
- Sharpening
- Edge detection
- Filtering is a neighbourhood operation

Neighborhoods
- Certain image processing operations involve processing an image insections, called blocks or neighborhoods, rather than processing the entire image at once.
- This is the case for:
* Filtering
* Histogram equalization for contrast enhancement
* Morphological functions

Edge detection
- This technique can be used to find the boundaries of objects within images.
- As well as segment and extract information like how items are in an image.
- Most of the shape information of an image is enclosed in edges.
- Edge detection works by detecting discontinuities in brightness
- A common edge detection algorithm is Sobel.

Sobel
# Import module and function
from skimage.filters import sobel

# Apply edge detection filter
edge_sobel = sobel(image_coins)

# Show original and resulting image to compare
plot_comparison(image_coins, edge_sobel, 'Edge with Sobel')

def plot_comparison(original, filtered, title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=Ture, sharey=True)
    ax1.imshow(original, smap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis('off')

Gaussian smoothing
- It will blur edges and reduce contrast
- It is used in other techniques like anti-aliasing filtering.
- The multichannel boolean parameter is set to True if the image is coloured, otherwise it needs to be set to False

Gaussian smoothing
# Import module and function
from skimage.filters import gaussian

# Apply edge detection filter
gaussian_image = gaussian(amsterdam_pic, multichannel=True)

# Show original and resulting image to compare
plot_comparison(amsterdam_pic, gaussian_image, 'Blurred with Gaussian filter')
'''

# Import the color module

# Import the filters module and sobel function

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")


# Import Gaussian filter

# Apply filter
gaussian_image = gaussian(building_image, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")


'''
Contrast enhancement
- The contrast of an image can be seen as the measure of its dynamic range, or the 'spred' of its histogram.
- It is the difference between the maximum and minimum pixel intensity in the image.
- An image of low contrast has small difference between its dark and light pixel values.
* It is usually skewed  either to the right (being mostly light)
* to the left (when its mostly dark)
* Around the middle (mostly gray)

Enhance contrast
- It can be enhance through contrast stretching which is used to stretch the histogram so the full range of intensity values of the image is filled.
- Histogram equalization spreads out the most frequent histogram intensity values using probability distribution.

Types of Histogram equalization
* The standard histogram equalization
* The adaptive histogram equalization
* Contrast limited Adaptive histogram equalization (CLAHE)

Histogram equalization
- It spreads out the most frequent intensity values

from skimage import exposure

# Obtain the equalized image
image_eq = exposure.equalize_hist(image)

# Show original and result
show_image(image, 'Original')
show_image(image_eq, 'Histogram equalized')

Adaptive Equalization
- This method computes several histograms, each corresponding to a distinct part of the image, and uses them to redistribute the lightness values of the image histogram.
- A type of this method is the Contrastive Limited Adaptive Histogram Equalization (CLAHE) which was developed to prevent over-amplification of noise that adaptive histogram equalization can give rise to.
* It does not take the global histogram of the entire image, but operates on small regions called tiles or neighborhoods.

CLAHE in scikit-image
- A clip limit is normalized between 0 and 1 (higher values give more contrast)

from skimage import exposure

# Apply adaptive Equalization
image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.3)

# Show original and result
show_image(image, 'Original')
show_image(image_adapteq, 'Adaptive equalized')
'''

# Import the required module

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq = exposure.equalize_hist(chest_xray_image)

# Show the resulting image
show_image(xray_image_eq, 'Resulting image')


# Import the required module

# Use histogram equalization to improve the contrast
image_eq = exposure.equalize_hist(image_aerial)

# Show the original and resulting image
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting image')


# Import the necessary modules

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(
    original_image, clip_limit=0.03)

# Compare the original image to the equalized
show_image(original_image)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')


'''
Transformations
Why transform images?
- Preparing images for classification Machine Learning models
- To Optimize and compress the size of images so it doesn't take long to analyze them
- To save images with same proportion

Rotating 
- Rotating images allows us to apply angles

Rotating clockwise
from skimage.transform import rotate

# Rotate the image 90 degrees clockwise
image_rotated = rotate(image, -90)

show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees clockwise')

Rotating anticlockwise
from skimage.transform import rotate

# Rotate the image 90 degrees anticlockwise
image_rotated = rotate(image, 90)

show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees anticlockwise')

Rescaling
- This operation resizes an image by a given scaling factor.
- setting an anti-aliasing boolean parameter to True specifies if applying a filter to smooth the image prior to down-scaling

Downgrading
from skimage.transform import rescale

# Rescale the image to be 4 times smaller
image_rescaled = rescale(image, 1/4, anti_aliasing=True, multichannel=True)

show_image(image, 'Original')
show_image(image_rescaled, 'Rescaled image')

Aliasing in digital images
- In a digital image, aliasing is a pattern or a rippling effect. 
- It makes the image look like it has waves or ripples radiating from a certain portion.
- It happens because the pixelation of the image is poor

Resizing
- It is used for making images match a certain size.
- It has the same purpose as rescale, but allows to specify an output image shape instead of a scaling factor.

from skimage.transform import resize

# Height and width to resize
height = 400
width = 500

# Resize image
image_resized = resize(image, (height, width), anti_aliasing=True)

show_image(image, 'Original image')
show_image(image_resized, 'Resized image')

Resizing proportionally
from skimage.transform import resize

# Set proportional height so its 4 times its size height = image.shape[0] / 4
width = image.shape[1] / 4

# Resize image
image_resized = resize(image, (height, width), anti_aliasing=True)

show_image(image, 'Original image')
show_image(image_resized, 'Resized image')
'''

# Import the module and the rotate and rescale functions

# Rotate the image 90 degrees clockwise
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4,
                           anti_aliasing=True, multichannel=True)

# Rescale without anti aliasing
rescaled_without_aa = rescale(
    rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")


# Import the module and function to enlarge images

# Import the data module

# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = rescale(
    rocket_image, 3, anti_aliasing=True, multichannel=True)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "3 times enlarged image")


# Import the module and function

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width), anti_aliasing=True)

# Show the original and resized image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')


'''
Morphology
Morphological filtering
- It tries to remove the imperfections by accounting for the form and structure of the objects in the image.
- They are espcially suited to binary images
* can extend to grayscale images

Morphological operations
- Basic morphological operations are dilation and erosion.
- Dilation adds pixels to the boundaries of objects in an image
- Erosion removes pixels on object boundaries
- The number of pixels added or removed from the objects in an image depends on the size and shape of a structuring element used to process the image.
* The structuring element is a small binary image used to probe the input image.
A - The structuring element fits the image
B - The structuring elements hits (intersects) the image
C - The structuring element neither fits, nor hits the image
- The dimensions specify the size of the structuring element

Shapes in scikit-image
from skimage import morphology

square = morphology.square(4) -> in

[   [1 1 1 1]
    [1 1 1 1]
    [1 1 1 1]
    [1 1 1 1]   ] -> out

rectangle = morphology.rectangle(4, 2) -> in

[   [1 1]
    [1 1]
    [1 1]
    [1 1]   ] -> out

Erosion in scikit-image
- if selem (structuring element) is not set, the default function will use cross-shaped structured element.

from skimage import morphology

# Set structuring element to the rectangular-shaped
selem = rectangle(12, 6)

# Obtain the erosed image with binary erosion
eroded_image = morphology.binary_erosion(image_horse, selem=selem)

# Show result
plot_comparison(image_horse, eroded_image, 'Erosion')

# Binary erosion with default selem
eroded_image = morphology.binary_erosion(image_horse)

# Show result
plot_comparison(image_horse, eroded_image, 'Erosion')

Dilation in scikit-image
from skimage import morphology

# Obtain dilated image, using binary dilation
dilated_image = morphology.binary_dilation(image_horse)

# Show result
plot_comparison(image_horse, dilated_image, 'Dilation')
'''

# Import the morphology module

# Obtain the eroded shape
eroded_image_shape = morphology.binary_erosion(upper_r_image)

# See results
show_image(upper_r_image, 'Original')
show_image(eroded_image_shape, 'Eroded image')


# Import the module

# Obtain the dilated image
dilated_image = morphology.binary_dilation(world_image)

# See results
show_image(world_image, 'Original')
show_image(dilated_image, 'Dilated image')


''' Image restoration, Noise, Segmentation and Contours '''

'''
Image restoration
Image reconstruction
- It is used for fixing damaged images
- For text removing
- For deleting Logos from images
- For removing small objects like tattoos

- Inpainting
* This is the process of reconstructing lost or deteriorated parts of images
* The reconstruction is suppposed to be performed in a fully automatic way by exploitig the information presented in the non-damaged regions of the image
- A mask image is simply an image where some of the pixel intensity values are zero and others are non-zero

Image reconstruction in scikit-image
from skimage.restoration import inpaint

# Obtain the mask
mask = get_mask(defect_image)

# Apply inpainting to the damaged image using the mask
restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)

# Show the resulting image
show_image(restored_image)

# Show the defect and resulting images 
show_image(defect_image, 'Image to restore')
show_image(restored_image, 'Image restored')

Mask
def get_mask(image):
    '' Creates mask with three defect regions ''
    mask = np.zeros(image.shape[:-1])

    mask[101:106, 0:240] = 1

    mask[152:154, 0:60] = 1
    mask[153:155, 60:100] = 1
    mask[154:156, 100:120] = 1
    mask[155:156, 120:140] = 1

    mask[212:217, 0:150] = 1
    mask[217:222, 150:256] = 1
    return mask
'''

# Import the module from restoration

# Show the defective image
show_image(defect_image, 'Image to restore')

# Apply the restoration function to the image using the mask
restored_image = inpaint.inpaint_biharmonic(
    defect_image, mask, multichannel=True)
show_image(restored_image)


# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[210:290, 360:425] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(
    image_with_logo, mask, multichannel=True)

# Show the original and logo removed images
show_image(image_with_logo, 'Image with logo')
show_image(image_logo_removed, 'Image with logo removed')


'''
Noise
- Noise is a result of errors in the image acquisition process that result in pixel values that do not reflect the true intensities of the real scene

Apply noise in scikit-image
# Import the module and function
from skimage.util import random_noise

# Add noise to the image
noisy_image = random_noise(dog_image)

# Show original and resulting image
show_image(dog_image)
show_image(noisy_image, 'Noisy image')

- Random noise in images is also known as 'salt and pepper'.
- The higher the resolution of the image, the longer it may take to eliminate the noise.

Denoising types
- Total variation (TV) Filtering
* This filter tries to minimize the total variation of the image.
* It tends to produce 'cartoon-like' images, i.e piecewise-constant images.

- Bilateral Filtering
* It smooths images while preserving edges
* It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels

- Wavelet denoising Filtering
- Non-Local means denoising Filtering

Denoising
Using Total variation filter denoising
- The greater the weight, the more denoising but it could make the image smoother.

from skimage.restoration import denoise_tv_chambolle

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, weight=0.1, multichannel=True)

# Show denoised image
show_image(noisy_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')

Bilateral filter
from skimage.restoration import denoise_bilateral

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(noisy_image, multichannel=True)

# Show original and resulting images
show_image(noisy_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')
'''

# Import the module and function

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show original and resulting image
show_image(fruit_image, 'Original')
show_image(noisy_image, 'Noisy image')


# Import the module and function

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, multichannel=True)

# Show the noisy and denoised images
show_image(noisy_image, 'Noisy')
show_image(denoised_image, 'Denoised image')


# Import bilateral denoising function

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image, multichannel=True)

# Show original and resulting images
show_image(landscape_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')


'''
Superpixels & segmentation
Segmentation
- Its goal is to partition images into regions, or segments, to simplify and/or change the representation into something more meaningful and easier to analyze.
- Thresholding is the simplest method of segmentation

Image representation
- Images are represented as a grid of pixels.

Superpixels
- This is the process of exploring more logical meanings in an image that's formed by bigger regions or grouped pixels.
- It is a group of connected pixels with similar colors or gray levels.
* These carry more meaning than their simple pixel grid counterparts.

Benefits of superpixels
- You can compute features on more meaningful regions.
- You can increase computational efficiency

Segmentation
- Supervised
* It is when some prior knowledge is used to guide the algorithm.
* Like the kind of thresholding in which we specify the threshold value ourselves

- Unsupervised
* It is when no prior knowledge is required.
* This algorithms try to subdivide images into meaningful regions automatically.
* The user may still be able to tweak certain settings to otain the desired output.

Unsupervised segmentation
- Simple Linear Iterative Clustering (SLIC)
* It segments the image using a ML algorithm called K-Means clustering.
* It takes in all the pixel values of the image and tries to separate them into a predefined number of sub-regions.

Unsupervised segmentation (SLIC)
# Import the modules
from skimage.segmentation import slic
from skimage.color import label2rgb

# Obtain the segments
segments = slic(image)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, image, kind='avg')

show_image(image)
show_image(segmented_image, 'Segmented_image')

More segments
# Import the modules
from skimage.segmentation import slic
from skimage.color import label2rgb

# Obtain the segmentation with 300 regions
segments = slic(image, n_segments = 300)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, image, kind='avg')

show_image(image)
show_image(segmented_image, 'Segmented_image')

- n_segments default value is 100 segments
'''

# Import the slic function from segmentation module

# Import the label2rgb function from color module

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments=400)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, face_image, kind='avg')

# Show the segmented image
show_image(segmented_image, "Segmented image, 400 superpixels")


'''
Finding contours
- A contour is a closed shape of points or line segments, representing the boundaries of these objects.
- It can be used to:
* Measure size
* Classify shapes
* Determine the number of objects

Binary images
- It can be obtained by applying thresholding or using edge detection.
- In binary image, the objects we wish to detect should be white, while the background remains black.

Find contour using scikit-image
PREPARING THE IMAGE
- Transform the image to 2D grayscale

# Make the image grayscale
image = color.rgb2gray(image)

- Binarize the image

# Obtain the thresh value
thresh = threshold_otsu(image)

# Applying thresholding
thresholded_image = image > thresh

- Use find_contours()
* This function finds the contour lines or joins point (pixels) of equal elevation or brightness in a 2D array above a given level value.

# Import the measure module
from skimage import measure

# Find contours at a constant value of 0.8
contours = measure.find_contours(thresholded_image, 0.8)

Constant level value
- The level value  varies between 0 and 1
- The closer to 1 the value, the more sensitive the method is to detecting contours, so more complex contours will be detected.
- We have to find the value that best detects the contours we care for.

The steps to spotting contours
from skimage import measure
from skimage.filters import threshold_otsu

# Make the image grayscale
image = color.rgb2gray(image)

# Obtain the optimal thresh value of the image
thresh = threshold_otsu(image)

# Apply thresholding and obtain binary image
thresholded_image = image > thresh

# Find contours at a constant value of 0.8
contours = measure.find_contours(thresholded_image, 0.8)

A contour's shape
Contours: list of (n, 2) - ndarrays.

for contour in contours:
    print(contour.shape) -> in

(433, 2)
(433, 2) --> Outer border
(401, 2)
(401, 2) --> Inner border
(123, 2)
(123, 2) --> Divisory lie of tokens
(59, 2)
(59, 2)
(59, 2)
(57, 2)
(57, 2)
(59, 2)
(59, 2)  Dots -> out

- The bigger the contour, the more points joined together and the wider the perimeter formed.
'''

# Import the modules

# Obtain the horse image
horse_image = data.horse()

# Find the contours with a constant level value of 0.8
contours = measure.find_contours(horse_image, 0.8)

# Shows the image with contours found
show_image_contour(horse_image, contours)


# Make the image grayscale
image_dice = color.rgb2gray(image_dice)

# Obtain the optimal thresh value
thresh = filters.threshold_otsu(image_dice)

# Apply thresholding
binary = image_dice > thresh

# Find contours at a constant value of 0.8
contours = measure.find_contours(binary, 0.8)

# Show the image
show_image_contour(image_dice, contours)


# Create list with the shape of each contour
shape_contours = [cnt.shape[0] for cnt in contours]

# Set 50 as the maximum size of the dots shape
max_dots_shape = 50

# Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

# Shows all contours found
show_image_contour(binary, contours)

# Print the dice's number
print("Dice's dots number: {}. ".format(len(dots_contours)))


''' Advanced Operations, Detecting Faces and Features '''

'''
Finding the edges with Canny
- Edge detection is extensively used when we want to divide the image into areas corresponding to different objects.
- Most of the shape information of an image is enclosed in edges.
- Representing an image by its edges has the advantage that the amount of data is reduced significantly while retaining most of the image information, like the shapes.

Edge detection
- The canny edge detection
* It is widely considered to be the standard edge detection method in image processing.
* It produces higher accuracy detecting edges and less execution time copared with Sobel algorithm.

from skimage.feature import canny

# Convert image to grayscale
coins = color.rgb2gray(coins)

# Apply Canny detector
canny_edges = canny(coins)

# Show resulted image with edges
show_image(canny_edges, 'Edges with Canny')

Canny edge detector
- The first step of this algorithm is to apply a gaussian filter inorder to remove noise in the image.
- sigma attribute is used to set the intensity of the gaussian filter to be applied in the image
* The lower the value of sigma, the less the gaussian filter effect is applied on the image, so it will spot more edges
* The higher the value of sigma, the more noise will be removed and the result is going to be less edgy image.
- default value of sigma is 1

# Apply Canny detector with a sigma of 0.5
canny_edges_0_5 = canny(coins, sigma = 0.5)

# Show resulted image with edges
show_image(canny_edges, 'Sigma of 1')
show_image(canny_edges_0_5, 'Sigma of 0.5')
'''

# Import the canny edge detector

# Convert image to grayscale
grapefruit = color.rgb2gray(grapefruit)

# Apply canny edge detector
canny_edges = canny(grapefruit)

# Show resulting image
show_image(canny_edges, "Edges with Canny")


# Apply canny edge detector with a sigma of 1.8
edges_1_8 = canny(grapefruit, sigma=1.8)

# Apply canny edge detector with a sigma of 2.2
edges_2_2 = canny(grapefruit, sigma=2.2)

# Show resulting images
show_image(edges_1_8, "Sigma of 1.8")
show_image(edges_2_2, "Sigma of 2.2")


'''
Right around the corner
Corner detection
- It is an approach used to extract certain types of features and infer the contents of an image.
- It is frequently used in: 
* Motion detection
* Image registration
* Video tracking
* Panorama stitching
* 3D modelling
* Object recognition
- It is basically detecting (one type of ) interest points in an image.

Points of interest
- Edges are a type of feature in images.
- Features are the points of interest which provide rich image content information
- Points of interest are points in the image which are invariant to rotation, translation, intensity and scale changes.
- There are different interes points such as
* Corners
* Edges

Corners
- It can be defined as the intersection of two edges
* It can also be a junction of contours

Harris corner detector
- It is a corner detector operator that is widely used in computer vision algorithms.

from skimage.feature import corner_harris

# Convert image to grayscale
image = rgb2gray(image)

# Apply the Harris corner detector on the image
measure_image = corner_harris(image)

# Show the Harris response image
show_image(measure_image)

# Finds the coordinates of the corners
coords = corner_peaks(corner_harris(image), min_distance=5)

print('A total of', len(coords), 'corners wer detected.') -> in

A total of 122 corners were found from measure response image. -> out

# Show image with marks in detected corners
show_image_with_detected_corners(image, coords)

Show image with contours
def show_image_with_corners(image, coords, title='Corners detected'):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
    plt.axis('off')
    plt.show()
'''

# Import the corner detector related functions and module

# Convert image from RGB-3 to grayscale
building_image_gray = color.rgb2gray(building_image)

# Apply the detector  to measure the possible corners
measure_image = corner_harris(building_image_gray)

# Find the peaks of the corners using the Harris detector
coords = corner_peaks(measure_image, min_distance=20, threshold_rel=0.02)

# Show original and resulting image with corners detected
show_image(building_image, "Original")
show_image_with_corners(building_image, coords)


# Find the peaks with a min distance of 10 pixels
coords_w_min_10 = corner_peaks(
    measure_image, min_distance=10, threshold_rel=0.02)
print("With a min_distance set to 10, we detect a total",
      len(coords_w_min_10), "corners in the image.")

# Find the peaks with a min distance of 60 pixels
coords_w_min_60 = corner_peaks(
    measure_image, min_distance=60, threshold_rel=0.02)
print("With a min_distance set to 60, we detect a total",
      len(coords_w_min_60), "corners in the image.")

# Show original and resulting image with corners detected
show_image_with_corners(building_image, coords_w_min_10,
                        "Corners detected with 10 px of min_distance")
show_image_with_corners(building_image, coords_w_min_60,
                        "Corners detected with 60 px of min_distance")


'''
Face detection
Face detection use cases
- To apply filters
- To auto focus in the face area
- Recommendations e.g tag a friend
- Automatically Blur faces for privacy protection.
- To recognize emotions later on

Detecting faces with scikit-image
# Import the classifier class
from skimage.feature import Cascade

# Load the trained file from the module root
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

- detect_multi_scale method searches for the object.
* It creates a window that will be moving through the image until it finds somthing similar to what is being searched for.
* Searching happens on multiple scales.
- The window will have a minimum size, to spot the small or far-away objects.
- It will also have a maximum size to find the larger objects in the image. 

Detecting faces
- The detector will return a dictionary of the coordinates of the box that contains the face.
- 'r' represents the row position of the top left corner of the detected window
- 'c' is the column position pixel
- 'width' is the width of the detected window
- 'height' is the height of the detected window

# Apply detector on the image
detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))

print(detected) -> in

Detected face: [{'r': 115, 'c': 210, 'width': 167, 'height': 167}] -> out

# Show image with detected face marked
show_detected_face(image, detected) 

Show detected faces
def show_detected_face(result, detected, title='Face image'):
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')

    for patch in detected:
        img_desc.add_patch(patches.Rectangle((patch['c'], patch['r']), patch['width'), patch['height'], fill=False, color='r', linewidth=2))
    plt.show()
'''

# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with min and max size of searching window
detected = detector.detect_multi_scale(
    img=night_image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))

# Show the detected faces
show_detected_face(night_image, detected)


# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with scale factor to 1.2 and step ratio to 1
detected = detector.detect_multi_scale(
    img=friends_image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))

# Show the detected faces
show_detected_face(friends_image, detected)


# Obtain the segmentation with default 100 regions
segments = slic(profile_image)

# Obtain segmented image using label2rgb
segmented_image = label2rgb(segments, profile_image, kind='avg')

# Detect the faces with multi scale method
detected = detector.detect_multi_scale(
    img=segmented_image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(1000, 1000))

# Show the detected faces
show_detected_face(segmented_image, detected)


'''
Real-world applications
Applications
- Turning images to grayscale before detecting edges / corners
- Reducing noise and restoring images
- Bluring faces detected
- Approximation of object's sizes

Privacy protection
# Import Cascade of classifiers and gaussian filter
from skimage.feature import Cascade
from skimage.filters import gaussian

# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect the faces
detected = detector.detect_multi_scale(img = image, scale_factor=1.2, step_ratio=1, min_size=(50, 50), max_size=(100, 100))

# For each detected face
for d in detected:
    # Obtain the face cropped from detected coordinates
    face = getFace(d)

    # Apply gaussian filter to extracted face
    gaussian_face = gaussian(face, multichannel=True, sigma = 10)

    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(image, gaussian_face)

def getFace(d):
    '' Extracts the face rectangle from the image using the coordinates of the detected.''
    # x and y starting points of the face rectangle
    x, y = d['r'], d['c']

    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']

    # Extract the detected face
    face = image[x:width, y:height]
    return face

def mergeBlurryFace(original, gaussian_image):
    # X and Y starting points of the face rectangle
    x, y = d['r'], d['c']

    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']

    original[ x:width, y:height ] = gaussian_image
    return original

- NB*: The classifier was only trained to detect the front side of faces, not profile faces.
* You can train the classifier with xml files of profile faces, that can be found available online.
* e.g some provided by the OpenCv image processing library.
'''

# Detect the faces
detected = detector.detect_multi_scale(
    img=group_image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(100, 100))

# For each detected face
for d in detected:
    # Obtain the face rectangle from detected coordinates
    face = getFaceRectangle(d)

    # Apply gaussian filter to extracted face
    blurred_face = gaussian(face, multichannel=True, sigma=8)

    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(group_image, blurred_face)
show_image(resulting_image, "Blurred faces")


# Import the necessary modules

# Transform the image so it's not rotated
upright_img = transform.rotate(damaged_image, 20)

# Remove noise from the image, using the chambolle method
upright_img_without_noise = denoise_tv_chambolle(
    upright_img, weight=0.1, multichannel=True)

# Reconstruct the image missing parts
mask = get_mask(upright_img)
result = inpaint.inpaint_biharmonic(
    upright_img_without_noise, mask, multichannel=True)

show_image(result)
