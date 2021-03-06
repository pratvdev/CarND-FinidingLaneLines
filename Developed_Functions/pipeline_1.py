#This function reads a color image and highlights lane markings in segments

def pipeline(image):     
    gray_img = grayscale(image)# converting color image to gray scale     
       
    blur_gscale =gaussian_blur(gray_img, 5)#blurring gray scale image with gaussian blur   
    cannyOut = canny(blur_gscale, 75, 150) # finding edged using Canny edge algorithm    

    imshape = image.shape 
    vertices_left = np.array([[(100,imshape[0]) , (420,340) ,(450 , 340), (290 , imshape[0])]] , dtype=np.int32) # vertices for the left polygon we want to draw on the image
    vertices_right = np.array([[(650 , imshape[0]), (520,340), (550,340), (940 , imshape[0])]] , dtype=np.int32) # vertices for the right polygon we want to draw on the image
    masked_edges_left = region_of_interest(cannyOut, vertices_left)
    masked_edges_right = region_of_interest(cannyOut, vertices_right)
    
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10   # maximum gap in pixels between connectable line segments    

    # Using Hough transform on the image with edges detected
    hough_lines_left = hough_lines(masked_edges_left, rho, theta, threshold, min_line_length, max_line_gap)
    hough_lines_right = hough_lines(masked_edges_right, rho, theta, threshold, min_line_length, max_line_gap)
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on 

    draw_lines(line_image, hough_lines_left)
    draw_lines(line_image, hough_lines_right)

    weighted_img(image, line_image)		   
       
    # drawing output lines on the original image
    lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0)
    
    # return type is the output image with highlighted lane markings
    return lines_edges

