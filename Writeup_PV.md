**Finding Lane Lines on the Road**

This project aims at finding lane marking on a few images and videos shared by Udacity.

The goals / steps of this project are the following:
* Create a function which reads images and plots lane markings on the image.
* Written report on the procedure of algorithm development.

#github URL: https://github.com/pratvdev/CarND-FinidingLaneLines.git

#Output Images: test_images_output/

#Output Videos: Outputs/

#Python functions: Developed_Functions/ 

#Python Notebook: Project1.ipynb


Pipeline development procedure:

1. Read an image in color format.
2. Convert this image into gray scale.
3. Blur this image using Gaussian Blur.
4. Find the Canny Edges using the computer vision function Canny.
5. Define regions of interest i.e. the lane marking areas on the images.
6. Find the lane marking in the regions of interest using Hough transform.
7. Use the output line segments from Hough transform to draw lines on the image (for segmented)
8. Try to extrapolate from the line segments from Hough transfor to draw lines on the image (for straight continuous lines)
9. Use the algorithm to work on videos.


### Detailed Steps:

### function pipeline(image) till step 11 (available in github link pipeline.py)
* Step 1: I created a small script to read in the test images from the path folder test_images and test my algorithm for segmented lane marking highlights.

* Step 2: Created an output folder test_images_output (available on the github link) to save my processed images. An example image reference is given below.

* Step 3: Defind a function pipeline which takes a color image as input and furthur process it to identify the segmented lane markings.

* Step 4: Used the function grayscale in the notebook to convert the image to gray scale.

* Step 5: Used the function gaussian_blur to blur the gray scale image from step 4.

* Step 6: Used the function canny to find the canny edges on the blurred image.

* Step 7: Created two regions of interest. One on the left side and the other on the right side to conver just the lane markings. Both the polygons were quadrilaterals.

* Step 8: Calculated the Hough lines using the function hough_lines for bott polygons defined in Step 7.

* Step 9: Created a blank image to draw lines for left and right lane markings. Used draw_lines function to do this.

* Step 10: Next combined the original image and the image from step 9 using the function weighted_img.

* Step 11: Tuned the polygon vertices and hough lines parameters till a desirable output was achieved. Image references till step 11 are given below.  

[image1]: ./test_images_output/solidWhiteCurve.jpg "Color"

[image2]: ./test_images_output/solidYellowLeft.jpg "Color"

### Next steps involve changinf the developed function draw continous straight lines on the images instead of segmented lines. For this the process changed a little from step 9.
### The changed steps are in the function process_image(image). This function is in the file process_image(image) in github link.

* Step 12: From the Hough lines output in Step 9, i segregated the x and y values and stored them in two different arrays for further computation. I did this procedure for both left and right regions of interest.

* Step 13: After segregating i used the polyfit command with poly1d from numpy to get the line equations in the form of y=mx+b. This process was again done for both the regions of interest.

* Step 14: I did not use the functions draw_lines or weighted_image to further process the image but included the functionality in thesame function process_image(image). For this i used the computer vision line command to draw lines. The parameter for x1,y1,x2,y2 changed this way. For left line, x1 is zero and i calculated y1 from using x=0 in the equation that i got from step 13. For x2, the value used is the highest value in the x array and corresponding y value is calculated from the line equation from Step 13. For right line, the highest value for x for the image is taken as x1 and corresponding y1 value is calculated. For x2 the minimum value from the segregated x array for right lane is taken and corresponding y2 value is calculated.

* Step 15: After drawing these lines, they are combined with the original image to show continous straight lines on the image. 


### Shortcomings

After successfully getting good results in outputs white.mp4 and yellow.mp4, when the algorithm was used on the challenge.mp4 it did not work properly.
The algorithm works well from straight roads but not curved roads.

### Improvements
Can improve the algorithm to work for curved roads and also make the region of interests more generic than hard coded numbers.
