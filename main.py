import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2



############################################################
# GLOBAL STRINGS
############################################################
# for main program panel 
PANEL_MESSAGE =             "\t1- (Part 1): Distance between vectors (euclidean distance)\n"+\
                            "\t2- (Part 2-1): Plot sin(x), cos(x), tan(x), cotan(x)\n"+\
                            "\t3- (Part 2-2): Cosine similarity between sin(x) and cos(s), tan(x) and cotan(x)\n"+\
                            "\t4- (Part 2-3): Difference between sin and cos, tan and cotan points\n"+\
                            "\t5- (Part 3): Image processing\n"+\
                            "\t6- (Part 4): Chess board similarity\n"+\
                            "\t7- (Part 5): Advanced image processing\n"+\
                            "\t0- Exit "
                            
SWITCH_MESSAGE =            "\n Plz select your function: "
EMPTY_INPUT =               "\n Empty input !!!"
INVALID_MESSAGE =           "\n Invalid input !!!"
RETURN_MESSAGE =            "\n Press any key to return panel ..."
TRY_AGAIN_MESSAGE =         "\n Press any key to try again ..."
EXIT_MESSAGE =              "<or 0 for exit>"


############################################################
## clear_screen
############################################################
def clear_screen():
    '''
    Code to clear running screen
    '''    
    # for mac and linux(here, os.name is 'posix')
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        # for windows platfrom
        _ = os.system('cls')

############################################################
## check_vector_hasNumberElenments
############################################################
def check_vector_hasNumberElenments(vector):
    '''
    check each element of vector is a number or not ?!
    ''' 
    for number in vector:
        try:
            float(number)
        except:
            return False
    return True


############################################################
## get_vectors
############################################################
def get_vectors(numberOfVectors, numberOfEelements_inVectors):
    '''
    get vector from user
    '''    
    print("\n Plz Enter " + str(numberOfVectors) + " vector; " +\
        "(Each vector seperated by «,» and " + str(numberOfVectors) + " element in every vector)")
    result_vector = []
    i = 0
    while i != numberOfVectors:
        tmp_input = input("\n vector " + str(i+1) + ": ")
        tmp_splited = tmp_input.split(",")
        tmp_vector = [x.strip() for x in tmp_splited]

        if len(tmp_vector) == numberOfEelements_inVectors and \
                check_vector_hasNumberElenments(tmp_vector) == True :
            result_vector.extend([float(x) for x in tmp_vector])
            i += 1
        else :
            input(INVALID_MESSAGE + TRY_AGAIN_MESSAGE)

    return result_vector


############################################################
## euclidean_distance
############################################################
def euclidean_distance(vector_1, vector_2):
    '''
    calculating euclidean distance
    '''  
    # subtracting vector 
    temp_vector = vector_1 - vector_2 
    
    # doing dot product for finding sum of the squares 
    sum_sq = np.dot(temp_vector.T, temp_vector) 
    
    # Doing squareroot and return Euclidean distance 
    return (np.sqrt(sum_sq)) 



############################################################
# SWITCH CLASS
############################################################ 
class Switch(object):
    ''' 
        Switch class: it represents switching between parts of panel.
    '''

    def select(self, i):
        ''' 
            Get a number and represents part of panel .
            Parameters: i - type: string - a number that used to select one part of the panel.
        ''' 
        method_name = 'number_' + str(i)
        method = getattr(self, method_name, lambda :input(INVALID_MESSAGE + RETURN_MESSAGE))
        return method()


    '''
    <Each method of this class represents one part of panel. And any of them help user 
     to get inputs and doing some thing.>
    '''
    def number_1(self):
        while True:
            clear_screen()
            print("\n\t *** Vector distance ***")
            numberOfVectors_userInput = input("\n Plz Enter number of vectors " +\
                                             EXIT_MESSAGE + ": ") 
            
            if numberOfVectors_userInput == "0":
                return
            elif numberOfVectors_userInput == "" or numberOfVectors_userInput.isdigit() == False:
                input(INVALID_MESSAGE + TRY_AGAIN_MESSAGE)
            else:
                numberOfEelements_inVectors_userInput = input("\n Plz Enter the number of elements in the vectors " +\
                                                             EXIT_MESSAGE + ": ")

                if numberOfEelements_inVectors_userInput == "0":
                    return
                elif numberOfEelements_inVectors_userInput == "" or numberOfEelements_inVectors_userInput.isdigit() == False:
                    input(INVALID_MESSAGE + TRY_AGAIN_MESSAGE)
                else:
                    numberOfVectors = int(numberOfVectors_userInput)
                    numberOfEelements_inVectors = int(numberOfEelements_inVectors_userInput)
                    vectorsArray = get_vectors(numberOfVectors, numberOfEelements_inVectors)

                    np_vectorsArray = np.array(vectorsArray)
                    np_vectorsArray = np_vectorsArray.reshape(numberOfVectors, numberOfEelements_inVectors)
                    print("\n Vectors that generated by user: ", np_vectorsArray)

                    np_randomVectorsArray = np.random.random(numberOfEelements_inVectors)
                    print("\n Random vector that generated with numpy: ", np_randomVectorsArray)

                    minimum_distance = sys.maxsize
                    for vector in np_vectorsArray:
                        tmp_min = euclidean_distance(vector, np_randomVectorsArray)
                        if tmp_min < minimum_distance:
                            minimum_distance = tmp_min
                    
                    print("\n ==> The minimum distance is: " + str(minimum_distance))
                    return input(RETURN_MESSAGE)
                    

    def number_2(self):
        
        clear_screen()
        print("\n\t *** Plot sin(x), cos(x), tan(x), cotan(x) ***")
        input("\n ==> Press any key to show plot: " )
        
        x1 = np.arange(0, (2 * np.pi), 0.1)
        x2 = np.arange(-(np.pi/2), +(np.pi/2), 0.1)
        fig, axs = plt.subplots(2,2)
        subplots_dict = [{ "i":0, "j":0, "x":x1, "y":np.sin(x1), "ylim": [-1.5, 1.5],
                           "y_label": "sin(x)", "x_label": "x values from 0 to 2pi"}, 
                         { "i":0, "j":1, "x":x1, "y":np.cos(x1), "ylim": [-1.5, 1.5],
                           "y_label": "cos(x)", "x_label": "x values from 0 to 2pi"},
                         { "i":1, "j":0, "x":x2, "y":np.tan(x2), "ylim": [-3, 3],
                           "y_label": "tan(x)", "x_label": "x values from -pi/2 to pi/2"},
                         { "i":1, "j":1, "x":x2, "y":1/np.tan(x2), "ylim": [-3, 3],
                           "y_label": "cotan(x)", "x_label": "x values from -pi/2 to pi/2"}]
        
        for subplot in subplots_dict:
            axs[subplot["i"], subplot["j"]].grid(True, which='both')
            axs[subplot["i"], subplot["j"]].plot(subplot["x"], subplot["y"])
            axs[subplot["i"], subplot["j"]].set_ylim(subplot["ylim"])
            axs[subplot["i"], subplot["j"]].set_xlabel(subplot["x_label"])
            axs[subplot["i"], subplot["j"]].set_ylabel(subplot["y_label"])
           
        fig.tight_layout()       
        plt.show()
        
        return input(RETURN_MESSAGE)


    def number_3(self):
        clear_screen()
        print("\n\t *** Cosine similarity between sin(x) and cos(s), tan(x) and cotan(x) ***")
        print("\n First we have to calculate sin and cos for x values from 0 to 2pi: ")
        
        x = np.arange(0, (2 * np.pi), 0.1)
        print("\n ==> x values from 0 to 2pi = ", x)
                
        y_sin = np.sin(x)
        y_cos = np.cos(x)

        print("\n ==> sin(x) = ", y_sin)
        print("\n ==> cos(x) = ", y_cos)

        cosine_similarity = np.dot(y_sin, y_cos)/np.dot(np.linalg.norm(y_sin), np.linalg.norm(y_cos))
        print("\n Now we calculate Cosine similarity with (sin(x) , cos(x)) as input: ")
        print("\n ==> Cosine similarity(sin(x) , cos(x)) = ", cosine_similarity)

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        print("\n First we have to calculate tan and cotan for x values from -pi/2 to pi2: ")
        
        x = np.arange(-(np.pi/2), (np.pi/2), 0.1)
        print("\n ==> x values from -pi/2 to pi2 = ", x)
                
        y_tan = np.tan(x)
        y_cotan = 1/np.tan(x)
        print("\n ==> tan(x) = ", y_tan)
        print("\n ==> cotan(x) = ", y_cotan)

        cosine_similarity = np.dot(y_tan, y_cotan)/np.dot(np.linalg.norm(y_tan), np.linalg.norm(y_cotan))
        print("\n Now we calculate Cosine similarity with (tan(x) , cotan(x)) as input: ")
        print("\n ==> Cosine similarity(tan(x) , cotan(x)) = ", cosine_similarity)
        
        return input(RETURN_MESSAGE)


    def number_4(self):
        clear_screen()
        print("\n\t *** Difference between sin and cos, tan and cotan points ***")
        print("\n First we have to put each x and sin(x) in vectors with two element in each like [x1, sin(x1)]")
        
        x = np.arange(0, (2 * np.pi), 0.1)

        y_sin = np.sin(x)
        sin_vector = np.array((x, y_sin)).T
        print("\n ==> sin_vector = ", sin_vector)
        

        print("\n Then we do as same for x and cos(x) like [x1, cos(x1)]")

        y_cos = np.cos(x)
        cos_vector = np.array((x, y_cos)).T
        print("\n ==> cos_vector = ", cos_vector)
                

        difference_vector = np.subtract(sin_vector, cos_vector)
        print("\n Now we subtract elements of two vector from each other: ")
        print("\n ==> difference_vector(sin_vector - cos_vector) = ", difference_vector)

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        print("\n First we have to put each x and tan(x) in vectors with two element in each like [x1, tan(x1)]")
        
        x = np.arange(-(np.pi/2), (np.pi/2), 0.1)

        y_tan = np.tan(x)
        tan_vector = np.array((x, y_tan)).T
        print("\n ==> tan_vector = ", tan_vector)
        

        print("\n Then we do as same for x and tan(x) like [x1, tan(x1)]")

        y_cotan = 1/np.tan(x)
        cotan_vector = np.array((x, y_cotan)).T
        print("\n ==> cotan_vector = ", cotan_vector)
                

        difference_vector = np.subtract(tan_vector, cotan_vector)
        print("\n Now we subtract elements of two vector from each other: ")
        print("\n ==> difference_vector(tan_vector - cotan_vector) = ", difference_vector)
        
        return input(RETURN_MESSAGE)

 
    def number_5(self):
        if not os.path.exists('Data\\Part3\\'):
            os.makedirs('Data\\Part3\\')
        clear_screen()
        print("\n\t *** Image processing ***")
        print("\n First we have load image.jpg from Data folder")
        
        image = Image.open('Data\\image.jpg')
        np_imageArray = np.array(image)

        np_imageArray_float64 = np.array(image, np.float64)
        print("\n ==> A: imageArray in float64 dataType = ", np_imageArray_float64.dtype)

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        np_blueImageArray = np_imageArray.copy()
        np_blueImageArray[:, :, (0, 1)] = 0
        blueImage = Image.fromarray(np_blueImageArray)
        blueImage.save('Data\\Part3\\blueImage.jpg')
        print("\n ==> B: change image to fully blue color and save it in Data\\Part3 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        nrows , ncols, nccolor = np_imageArray.shape
        rowIndex = int(nrows/2)
        colIndex = int(ncols/2)
       
        np_image1 = np_imageArray[:rowIndex, :colIndex]
        np_image2 = np_imageArray[:rowIndex, colIndex:]
        np_image3 = np_imageArray[rowIndex:, :colIndex]
        np_image4 = np_imageArray[rowIndex:, colIndex:]
        Image.fromarray(np_image1).save('Data\\Part3\\image1.jpg')
        Image.fromarray(np_image2).save('Data\\Part3\\image2.jpg')
        Image.fromarray(np_image3).save('Data\\Part3\\image3.jpg')
        Image.fromarray(np_image4).save('Data\\Part3\\image4.jpg')
        
        print("\n ==> C: split image into 4 pieces and save them in Data\\Part3 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        np_negetiveImage1 = 255 - np_image1
        Image.fromarray(np_negetiveImage1).save('Data\\Part3\\negetiveImage1.jpg')
        print("\n ==> D: change image1 color table to negetive and save it in Data\\Part3 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        nrows , ncols, nccolor = np_negetiveImage1.shape
        np_trimNegetiveImage1 = np_negetiveImage1[int(nrows/2):, int(ncols/6)*5:]
        Image.fromarray(np_trimNegetiveImage1).save('Data\\Part3\\trimNegetiveImage1.jpg')
        print("\n ==> E: trim negetiveImage1 to small piece and save it in Data\\Part3 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )

        tmpImage = Image.open('Data\\Part3\\trimNegetiveImage1.jpg')
        imageFormat = tmpImage.format
        imageShape = tmpImage.size
        print("\n ==> F: trimNegetiveImage1 format: ", imageFormat, \
              "\n        trimNegetiveImage1 size(width, height): ", imageShape)

        print("###############################################################################")

        return input(RETURN_MESSAGE)


    def number_6(self):
        if not os.path.exists('Data\\Part4\\'):
            os.makedirs('Data\\Part4\\')
        clear_screen()
        print("\n\t *** Chess board similarity ***")
        input("\n First we have to generate and show two chess board with different height and width:" +\
              "\n ==> Press any key to show plot: " )
        
        mat1 = np.random.randint(0, 8, (8,18))
        mat2 = np.add.outer(range(8), range(8)) % 2 

        fig = plt.figure()
        ax0 = plt.subplot2grid((2,2),(0,0), colspan=2)
        ax1 = plt.subplot2grid((2,2),(1,0), colspan=1)
        
        ax0.imshow(mat1, cmap='Oranges', interpolation='nearest')
        ax1.imshow(mat2, cmap='Oranges', interpolation='nearest')
        
        fig.tight_layout()
        plt.show()
        

        print("\n ==> Now find similarity between each part of chess boards and save them with their similarity in Data\\Part4 folder.")
        print("\n     Wait a moment till end saving all pictures ... ")
        
        thresHoldMat = np.full((8, 8), 3)
        zeros = np.ones((8,8))
        
        nrows1, ncols1 = mat1.shape
        nrows2, ncols2 = mat2.shape

        mat_arr = []
        i = 0
        while i + ncols2 <= ncols1:
            mat_arr.append(mat1[:,i:ncols2+i])
            i += 1


        mat_res = []
        for mat in mat_arr:
            mat_res.append(np.less_equal(thresHoldMat, mat).astype(int))
        
        for i in range(len(mat_res)):
            fig = plt.figure()
            ax0 = plt.subplot2grid((2,2),(0,0), colspan=2)
            ax1 = plt.subplot2grid((2,2),(1,0), colspan=1)
            ax2 = plt.subplot2grid((2,2),(1,1), colspan=1)

            ax0.imshow(mat1, cmap='Oranges', interpolation='nearest')
            ax1.imshow(mat2, cmap='Oranges', interpolation='nearest')
            ax2.imshow(mat_res[i], cmap='Oranges', interpolation='nearest')
        
            fig.tight_layout()
            fig.savefig('Data\\Part4\\chessBoardsWithSimilarity' + str(i) + '.jpg')

                
        print("\n     All pictures saved successfully.")
        
        
        return input(RETURN_MESSAGE)


    def number_7(self):
        if not os.path.exists('Data\\Part5\\'):
            os.makedirs('Data\\Part5\\')
        clear_screen()
        print("\n\t *** Advanced image processing ***")
        print("\n First we have load image.jpg from Data folder")
        
        image = Image.open('Data\\image.jpg')
        np_imageArray = np.array(image)

        np_lowColorImageArray = np_imageArray // 128 * 128
        lowColorImage = Image.fromarray(np_lowColorImageArray)
        lowColorImage.save('Data\\Part5\\lowColorImage.jpg')
        print("\n ==> A: decrease image color and save it in Data\\Part5 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        angle = -45
        rotatedImage = lowColorImage.rotate(angle, expand=True)
        rotatedImage.save('Data\\Part5\\rotatedImage.jpg')
        print("\n ==> B: rotate lowColorImage 45 degree and save it in Data\\Part5 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        image2 = cv2.imread('Data\\Part5\\rotatedImage.jpg')
        flipedImage = cv2.flip(image2, 1)
        cv2.imwrite('Data\\Part5\\flipedImage.jpg', flipedImage)
        print("\n ==> C: flip rotatedImage and and save it in Data\\Part5 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        flipedImage2 = Image.open('Data\\Part5\\flipedImage.jpg')
        np_flipedImage2Array = np.array(flipedImage2)

        nrows , ncols, nccolor = np_flipedImage2Array.shape
        np_cropedImageArray = np_flipedImage2Array[int(nrows/5)*2:int(nrows/5)*3, int(ncols/5)*2:int(ncols/5)*3]
        cropedImage = Image.fromarray(np_cropedImageArray)
        cropedImage.save('Data\\Part5\\cropedImage.jpg')
        print("\n ==> D: crop flipedImage and save it in Data\\Part5 folder.")

        print("###############################################################################")
        input("\n ==> Press any key to continue ... " )
        
        tmp_image = np_imageArray.copy()
        tmp_nrows , tmp_ncols, tmp_nccolor = tmp_image.shape
        croped_nrows , croped_ncols, croped_nccolor = np_cropedImageArray.shape
      
        tmp_image[int(tmp_nrows/7)*2:int(tmp_nrows/7)*2 + croped_nrows,
                 int(tmp_ncols/5)*2:int(tmp_ncols/5)*2 + croped_ncols] = np_cropedImageArray
        combinedImage = Image.fromarray(tmp_image)
        combinedImage.save('Data\\Part5\\combinedImage.jpg')
        print("\n ==> E: combine cropedImage with orginal image and save it in Data\\Part5 folder.")
        print("###############################################################################")

        return input(RETURN_MESSAGE)

       

if __name__ == "__main__":
    s = Switch()
    while True :
        clear_screen()
        print()
        print(PANEL_MESSAGE)
        switch_input = ""
        switch_input = input(SWITCH_MESSAGE)
        if switch_input == "0":
            clear_screen()
            print("\n\t *** Good Bye ***")
            break
        elif switch_input == "":
                input(EMPTY_INPUT + TRY_AGAIN_MESSAGE)
        else:
            s.select(switch_input)

