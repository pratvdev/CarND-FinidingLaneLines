# This function reads the images and prints a hightligted straight lines on the lane markings

def process_image(image):
    gray_img = grayscale(image)# converting color image to gray scale     
       
    blur_gscale =gaussian_blur(gray_img, 5)#blurring gray scale image with gaussian blur   
    cannyOut = canny(blur_gscale, 75, 150) # finding edged using Canny edge algorithm    

    imshape = image.shape
    
    mid_y = (0.6)*imshape[0] #top y corordinate for right and left polygons
    
    left_lbottomx = (0.1)*imshape[1] # left bottom x coordinate for left polygon
    left_rbottomx = (0.3)*imshape[1] #right bottom x coordinate for left polygon
    left_lmidx = (0.43)*imshape[1] # left mid x coordinate for left polygon
    left_rmidx = (0.47)*imshape[1] # right mid x coordinate for left polygon    
    
    right_lbottomx = (0.7)*imshape[1] #left bottom x for right polygon
    right_rbottomx = (0.98)*imshape[1] #right bottomx for right polygon
    right_lmidx = (0.54)*imshape[1] # left mid x coordinate for right polygon
    right_rmidx = (0.58)*imshape[1] # right mid x coordinate for right polygon
    
    
    # vertices for the left polygon we want to draw on the image
    vertices_left = np.array([[(left_lbottomx,imshape[0]) , (left_lmidx,mid_y) ,(left_rmidx , mid_y), (left_rbottomx , imshape[0])]] , dtype=np.int32)
    # vertices for the right polygon we want to draw on the image
    vertices_right = np.array([[(right_lbottomx , imshape[0]), (right_lmidx,mid_y), (right_rmidx,mid_y), (right_rbottomx , imshape[0])]] , dtype=np.int32)
    
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
    
    left_x_array = []
    left_y_array = []
    
    # segregating x and y values from the hough lines using for loops
    for line in hough_lines_left:
        for x1,y1,x2,y2 in line:
            left_x_array.append(x1)
            left_x_array.append(x2)
            left_y_array.append(y1)
            left_y_array.append(y2)
            
    right_x_array = []
    right_y_array = []
    for line in hough_lines_right:
        for x1,y1,x2,y2 in line:
            right_x_array.append(x1)
            right_x_array.append(x2)
            right_y_array.append(y1)
            right_y_array.append(y2)
            
    # Using numpy polyfit function along with poly1d to get the line equation y=mx+b.
    # The euqation in left_line and right_line helps us draw straight line on the image for lane markings
    
    left_lines_coef = np.polyfit(left_x_array , left_y_array ,1)
    left_line = np.poly1d(left_lines_coef)
    
    right_lines_coef = np.polyfit(right_x_array , right_y_array ,1)
    right_line = np.poly1d(right_lines_coef)    
    
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # drawing output lines on a blank image
    cv2.line(line_image,(0,int(left_line(0))),(max(left_x_array),int(left_line(max(left_x_array)))),(0,255,0),10)
    cv2.line(line_image,(imshape[1],int(right_line(imshape[1]))),(min(right_x_array),int(right_line(min(right_x_array)))),(0,255,0),10)

    # drawing output lines on the original image
    lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0)
    
    # return type is the output image with highlighted lane markings
    return lines_edges

