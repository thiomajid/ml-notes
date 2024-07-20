> Computer vision is the science and technology of making machines see. It involves the development of theoretical and algorithmic methods to acquire, process, analyze, and understand visual data, and to use this information to produce meaningful representations, descriptions, and interpretations of the world
\- _Forsyth & Ponce, Computer Vision: A Modern Approach_

## Fundamentals
An image is a *visual representation* of an object, a scene, a person, or even a concept. An image is also an *n-dimensional function*.
In the case of a 2D image grayscale image, the function $f(x, y)$ which describes the image where $x, \text{ and } y$ are **spatial coordinates** (position in a physical space, in this case the 2D cartesian system) of a *pixel* is defined as:
$$
f: \mathbb{R}^{2}\rightarrow \mathbb{R}
$$

The value of the function $f$ at coordinates $(x_{i,}y_i)$ represent the *intensity* or *color* at that point. This intensity gives us the notion of *dark* and *light*.

>[!info]
>3D images are called **volumetric images** and here the function takes as input a triplet $(x_{i,}y_{i,}z_i)$ defining the position of a **voxel** (volumetric element).

Even though images are mainly represented using matrices, they can also be represented as *graphs* where each node is a coordinate, and the edges are the neighboring coordinates. In other words, graph algorithms can be applied to images too.

Images are data points that contain a lot of **spatial information**. 
A video can be seen as an image with an associated *time* component. Thus, the function will be defined as : $f(x,y,t)$.

### Imaging
The core of digital image formation is the function $f(x,y)$, which is determined by the illumination source $i(x,y)$, and the reflectance $r(x,y)$ from the scene.

Image restoration consists of applying the *inverse* degradation process through which an image underwent to regain the original image. In contrast, image enhancement aims to **improve the visual appearance** of an image.

When compressing data, a compression ratio of 10:1 for instance, indicates 90% of data redundancy. When compressing images, in particular 2D intensity arrays, the three main types of redundancies are: **coding redundancy, spatial and temporal redundancy and irrelevant information**.
Techniques like *RLE* can reduce spatial redundancy. 

**Spatial resolution** refers to the smallest distinguishable detail in an image and is often measured in line pairs per unit distance or pixels per unit distance. The meaningfulness of spatial resolution is context-dependent, varying according to the spatial units used.

 **Intensity resolution** relates to the smallest detectable change in intensity level and is often limited by the hardware’s capabilities. It’s quantized in binary increments, such as 8 bits or 256 levels.

## What is computer vision ?

Image understanding is the process of making sense of the content of an image. It is defined at three levels:
- **Low-level processes:** image sharpening and changing contrast, etc...Image in, image out.
- **Mid-level processes:** segmentation, object detection and description. We are interested in *attributes* associated with the image.
- **High-level processes:** objection recognition, scene description, image reconstruction or image-to-text.

### Pre-processing for Computer Vision Tasks
In digital image processing, operations on images are diverse and can be categorized into: *logical* (AND, OR, XOR, etc... Think of pixel inversion in gray-scale images using the NOR op), *statistical, geometrical, mathematical, transform ops*. 

### Feature description
Features are attributes of the instances learned by the model to be later used to recognize new instances.
