def process_image(image):
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
    
    left_x_array = []
    left_y_array = []
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


