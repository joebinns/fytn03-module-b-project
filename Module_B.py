from PIL import Image
import numpy as np
import time
from copy import deepcopy
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initiate timer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
startedRunning = time.perf_counter()


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~ User decisions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
# Select the image and mask from available options.
# Automatically determine the masked image from this information.
# Choose if to apply averaging.

def ValidatedInput(inputQueue, answerRange, dtype = int):
    while True: # Check that the input is a number, and then check that the number is within the desired range for answers.
        try:
            theInput = dtype(input(inputQueue))
            while True: # Check that the input, which has already been established to be a number, is within the desired range for answers.
                if (theInput >= answerRange[0] and theInput <= answerRange[1]):
                    break # Escape the while loop if the input is within the range.
                else:
                    print("Input was not within the range of the options. Please re-enter.")
                    theInput = dtype(input(inputQueue))                    
            break # Escape the while loop if the input is interpreted as a NUMBER within the RANGE.               
        except:
            print("Input could not be interpreted as a number. Please re-enter. HELP: Type the desired number ONLY and press enter.")
    return theInput

def QuestionAndAnswer(question, options):
    print(question)
    for i in range(0, len(options)):
        print(str(i+1) + ": " + options[i]) 
    choice = ValidatedInput("Make your selection: ", [1, len(options)]) - 1
    return(choice)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User Decisions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
options = []
answers = []

# Choose oimage!
oimageFilePaths = [r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB.jpg']
oimagePathChoice = oimageFilePaths[QuestionAndAnswer("Select the file path of the original image.", oimageFilePaths)]
print()
options.append("oimagePathChoice: ")
answers.append(oimagePathChoice)

# Choose mask (and therefore also the masked image)
maskFilePathsAll = {
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS.jpg': [r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_RECTANGLES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_FLIES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_BIGFLIES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_TEXT.jpg'],
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB.jpg': [r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_RECTANGLES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_FLIES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_BIGFLIES.jpg', r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_TEXT.jpg']
        }
maskFilePaths = maskFilePathsAll[oimagePathChoice]
maskPathChoice = maskFilePaths[QuestionAndAnswer("Select the file path of the mask.", maskFilePaths)]
print()
options.append("maskPathChoice: ")
answers.append(maskPathChoice)

# Automatically determine the masked version of the image.
imageFilePathsGS = {
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_RECTANGLES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_RECTANGLED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_FLIES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_FLIED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_BIGFLIES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_BIGFLIED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_TEXT.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS_TEXTED.jpg'
        }
imageFilePathsRGB = {
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_RECTANGLES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_RECTANGLED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_FLIES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_FLIED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_BIGFLIES.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_BIGFLIED.jpg',
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_TEXT.jpg': r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB_TEXTED.jpg'
        }
imageFilePathsAll = {
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_GS.jpg': imageFilePathsGS,
        r'C:\Users\danie\Documents\Physics_Bachelor\Physics\FYTN03-Computational_Physics\FYTN03_Module_B_Project-main\LOTF_RGB.jpg': imageFilePathsRGB
        }

imagePathChoice = imageFilePathsAll[oimagePathChoice][maskPathChoice]
options.append("imagePathChoice: ")
answers.append(maskPathChoice)

'''---------- NEW --------------'''
ifRGBOptions = ["False", "True"]
ifRGB = ifRGBOptions[QuestionAndAnswer("If the image is RGB, convert it to grayscale?", ifRGBOptions)]
if (ifRGB == "True"):
    ifRGB = True
else:
    ifRGB = False
options.append("ifRGB: ")
answers.append(str(ifRGB))
print()
'''------------------------------'''

isAveragingOptions = ["False", "True"]
isAveraging = isAveragingOptions[QuestionAndAnswer("Apply colour-averaging to the mask using it's surroundings, prior to applying the solving method?", isAveragingOptions)]
if (isAveraging == "True"):
    isAveraging = True
else:
    isAveraging = False
options.append("isAveraging: ")
answers.append(str(isAveraging))
print()
    
# Choose solver!
solvers = ["SOR"]
solverChoice = solvers[QuestionAndAnswer("Select the solver to be used!", solvers)]
options.append("solverChoice: ")
answers.append(solverChoice)

NMutable = []
omegaMutable = []
SOROptions = {"Number of iterations (N) [1, 200]: ": [[1, 200], NMutable, int],
              "Relaxation (\u03C9) [0.01, 1.99]: ": [[0.01, 1.99], omegaMutable, float]
              }
solversAll = {
        "SOR": SOROptions
        }
for key in solversAll[solverChoice]:
    allowedRange = solversAll[solverChoice][key][0]
    var = solversAll[solverChoice][key][1]
    dtype = solversAll[solverChoice][key][2]
    var.append(ValidatedInput(key, allowedRange, dtype = dtype))
    options.append(key)
    answers.append(var[0])
N = NMutable[0]
omega = omegaMutable[0]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User Choices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for i in range(0, len(options)):
    print(str(options[i]) + str(answers[i]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print()


'''--------------------New-------------------------------------'''
''' ~~~~~~~~~~~~~~~~~~~~~~~~ RGB decomposition of image ~~~~~~~~~~~~~~~~~~~~~~~~~ '''
def RGB_decomp(RGB_canvas):
    '''
    Input(s):
        img {array}: full image, with all three colour pixels.
    Output(s):
        red_canvas {array}: array of the intensities of the red pixel, taking values [0, 255] corresponding to [black, red].
        green_canvas {array}: array of the intensities of the red pixel, taking values [0, 255] corresponding to [black, red].
        blue_canvas {array}: array of the intensities of the red pixel, taking values [0, 255] corresponding to [black, red].
    '''
    red_canvas = RGB_canvas[:,:,0]
    green_canvas = RGB_canvas[:,:,1]
    blue_canvas = RGB_canvas[:,:,2]
    return np.array(red_canvas, dtype = np.int16), np.array(green_canvas, dtype = np.int16), np.array(blue_canvas, dtype = np.int16)

''' ~~~~~~~~~~~~~~~~~~~~~ Check if greyscale ~~~~~~~~~~~~~~~~~~~~ '''
'''
    Since an image that appears to be grayscale may actually be storing some
    colour values (i.e. a digital colour photograph taken of a greyscale image
    may interpret some colour, such as a general hue). I will check that the image
    is truly grayscale here, and if it is found not to be, I will apply the 
    greyscale filter to the image in an image editing software, and reimport.
'''

def IsGreyScale(img_path): # Source: https://stackoverflow.com/a/23661373
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

''' ~~~~~~~~~~~~~~~~~ Import original image and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
oimgInPath = oimagePathChoice # path of the original, unmasked image.
oimgIm = Image.open(oimgInPath) # open image

RGB = False # RGB is set to False by default
if IsGreyScale(oimgInPath) == False and ifRGB == False:
    oRGBImgMat = np.array(oimgIm) # creates an array of the original image
    oRedImgMat, oGreenImgMat, oBlueImgMat = RGB_decomp(oRGBImgMat) # isolates the colours, red, green and blue of the image
    RGB = True 
else:
    oimgIm = oimgIm.convert('L') # converts image to grayscale.
    oimgMat = np.array(oimgIm, dtype = np.int16) # gets array of image.


''' ~~~~~~~~~~~~~~~~~ Import image and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
imgInPath = imagePathChoice # path of the image with mask applied.
imgIm = Image.open(imgInPath)
cpimgIm = Image.open(imgInPath)

# if soving in RGB
if RGB == True:
    RGBImgMat = np.array(imgIm) # creates an array of the original RGB image
    RedImgMat, GreenImgMat, BlueImgMat = RGB_decomp(RGBImgMat) # isolates the colours, red, green and blue of the image

# even if image is RGB, masks are found on grayscale image
imgIm = imgIm.convert('L') # converts image to grayscale.
imgMat = np.array(imgIm, dtype = np.int16) # gets array of image.

#canvasWidth, canvasHeight = imgIm.size #defines the matrix dims 
'''--------------------------------------------------------------------'''


''' ~~~~~~~~~~~~~~~~~~ Import mask and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
maskInPath = maskPathChoice # path of the mask which has been applied.
maskIm = Image.open(maskInPath)
maskIm = maskIm.convert('L') # converts mask image to grayscale.
maskMat = np.array(maskIm, dtype = np.int16) # gets array of mask image.

''' ~~~~~~~~~~~~~~~~~~~~~~ Break mask into submasks ~~~~~~~~~~~~~~~~~~~~~~~ '''
# Re-purposed queue fill algorithm source: https://www.algorithm-archive.org/contents/flood_fill/flood_fill.html
from queue import Queue

# Rough procedure:
# 1: Searching row by row for non-1 intensity.
# 2: When found,  create a sub-mask matrix (the size of the canvas).
# 3: Apply fill to the point, adding appropriate point to the sub-mask when enqueued.
# 4: Continue scanning the grid, only applying fill if the point isn't already included in any sub-mask.

def I(x, y, canvas = imgMat):
    '''
    Note: Origin of the image is treated as the top-left-most pixel, corresponding to [x,y] = [0,0].
    Input(s):
        x {int}: x coordinate of the pixel, in the range [0, imageWidth - 1].
        y {int}: y coordinate of the pixel, in the range [0, imageHeight - 1].
        canvas {nested array}: The matrix of i.e. the image.
    Output(s):
        canvas[y][x] {int}: The intensity of the pixel at coordinate (x, y), taking values [0, 255] corresponding to [black, white].
    '''
    return(canvas[y][x])

def WithinBounds(canvas, p):
    '''
    Input(s):
        canvas {nested array}: Matrix of image of concern.   
        p {length-2 array}: Coordinate of the pixel to check in form [x, y].
    Output(s):
        isWithinBounds {bool}: Whether or not p is within the bounds on the image.
    '''
    isWithinBounds = (p[0] >= 0 and p[0] < len(canvas[0])) and (p[1] >= 0 and p[1] < len(canvas))
    return isWithinBounds

def AlreadySubmasked(submasks, p):
    '''
    Input(s):
        submasks {array of nested arrays}: List of all the submasks matrices.   
        p {length-2 array}: Coordinate of the pixel to check in form [x, y].
    Output(s):
        isAlreadySubmasked {bool}: Whether or not p is already contained within any of the submasks.
    '''
    isAlreadySubmasked = False      
    for submask in submasks:
        if IsMask(submask[p[1]][p[0]]):
            isAlreadySubmasked = True
            break
    return isAlreadySubmasked

def IsMask(i):
    '''
    Input(s):
        i {int}: Pixel intensity. 
    Output(s):
        isMask {bool}: wether or not the pixel intensity qualifies as a mask or not (i.e. 0 is a mask, 255 is not).
    '''
    isMask = False
    if (i < 255 * 0.9): # i == 1 is a temporary fix. For some unknown reason, the imported mask matrix has values of 0 and 1 (on a scale of [0, 255]) for the black, whereas we expect only 0. 
        isMask = True
    return isMask

def FindNeighbours(canvas, submasks, p, allowDiagonalNeighbours = True):
    '''   
    Input(s):
        canvas {nested array}: The matrix of the mask. 
        submasks {array of nested arrays}: List of all the submasks matrices. 
        p {length-2 array}: Coordinate of the pixel to work from in the form [x, y].
    Output(s):
        neighbours {array of length-2 arrays}: A collection of all the appropriate neighbours (i.e. within bounds of canvas, not already contained in another submask, has intensity expected of mask).
    '''
    if (allowDiagonalNeighbours):
        possibleNeighbours = [
            [p[0],     p[1] + 1], # South
            [p[0] + 1,     p[1]], # East
            [p[0],     p[1] - 1], # West
            [p[0] -     1, p[1]], # North
            
            [p[0] + 1, p[1] + 1], # South-East
            [p[0] - 1, p[1] + 1], # South-West
            [p[0] + 1, p[1] - 1], # North-East
            [p[0] - 1, p[1] - 1] # North-West
            ]
    else:
        possibleNeighbours = [
            [p[0],     p[1] + 1], # South
            [p[0] + 1,     p[1]], # East
            [p[0],     p[1] - 1], # West
            [p[0] -     1, p[1]] # North
            ]
    # Exclude the neighbors that are out of bounds, already in another submask, or should not be coloured.
    neighbours = []
    for possibleNeighbour in possibleNeighbours:    
        if WithinBounds(canvas, possibleNeighbour):
            if (AlreadySubmasked(submasks, possibleNeighbour) == False):
                if IsMask(I(possibleNeighbour[0], possibleNeighbour[1], canvas = canvas)):
                    neighbours.append(possibleNeighbour)                   
    return neighbours

def QueueFill(canvas, submasks, p):
    '''
    Input(s):
        canvas {nested array}: The matrix of the mask. 
        submasks {array of nested arrays}: Empty list of all the submasks matrices. 
        p {length-2 array}: Coordinate of the initial pixel to work out from in the form [x, y].
    Output(s):
        submasks {array of nested arrays}: Completed list of all the submasks matrices.
    '''
    submask = np.full((len(canvas), len(canvas[0])), 255, dtype = int)
    submasks.append(submask)

    q = Queue()
    
    submask[p[1]][p[0]] = 0 # Setting the value of the start position to the new value.
    q.put(p) # Adding the start position to the queue.

    while not q.empty(): # While there remains neighbouring cells.
        currentLoc = q.get()
        neighbours = FindNeighbours(canvas, submasks, currentLoc, allowDiagonalNeighbours = True)
        for neighbour in neighbours:
            submask[neighbour[1]][neighbour[0]] = 0
            q.put(neighbour)
            
    return(submasks)

# Traverse through the mask matrix row-by-row, applying the 'detection-fill' algorithm to pixels which qualify as having the intensity of a mask, and have not already been included in a previous fill.
submasks = []
for y in range(0, len(imgMat)):
    for x in range(0, len(imgMat[0])):
        if (IsMask(I(x, y, canvas = maskMat))):
            if (AlreadySubmasked(submasks, [x, y]) == False):
                submasks = QueueFill(maskMat, submasks, [x, y])

# Traverse through the completed submasks, create new reduced submasks bounded by a rectangle surrounding all the mask pixels. 
def ReduceSubmasks(canvas, submasks, isAveraging):
    reducedSubmasks = []
    reducedSubmasksOrigins = []
    for submask in submasks:
        reducedSubmask = []  
        minX = len(canvas[0])
        minY = len(canvas)
        maxX = 0
        maxY = 0
        for y in range(0, len(canvas)):
            for x in range(0, len(canvas[0])):
                if (IsMask(I(x, y, canvas = submask))):
                    if (x < minX):
                        minX = x
                    elif (x > maxX):
                        maxX = x
                    if (y < minY):
                        minY = y
                    elif (y > maxY):
                        maxY = y                      
        if (isAveraging):
            minX -= 1
            minY -= 1
            maxX += 2
            maxY += 2
        else:       
            maxX += 1
            maxY += 1
        for y in range(minY, maxY):
            row = []
            for x in range(minX, maxX):
                row.append(I(x, y, canvas = submask))
            reducedSubmask.append(row)
        reducedSubmasks.append(reducedSubmask)
        reducedSubmasksOrigins.append([minX, minY])
    return(reducedSubmasks, reducedSubmasksOrigins)

reducedSubmasks, reducedSubmasksOrigins = ReduceSubmasks(imgMat, submasks, isAveraging)

''' # Printing submasks row by row (DEBUGGING, check the submask image files which are saved/displayed later.)
for submask in reducedSubmasks:
    print()
    for row in range(0,len(submask)):
        print(submask[row])
    print()
'''   

'''---------------------New-------------------'''
''' ~~~~~~~~~~~ Calculating the average from all the boundaries ~~~~~~~~~~~ '''
def CalculateAveraging():
    averages = []
    averagesRed = []
    averagesGreen = []
    averagesBlue = []
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]
        averageTally = 0 # average tally for gray
        averageTallyRed = 0 # average tally for red
        averageTallyGreen = 0 # average tally for green
        averageTallyBlue = 0 # average tally for blue
        pixelsAveraged = 0   
        for y in range(len(submask)):
            for x in range(len(submask[y])):
                if (IsMask(I(x, y, canvas = submask)) == False):
                    offX = x+reducedSubmasksOrigins[i][0]
                    offY = y+reducedSubmasksOrigins[i][1]
                    if RGB == True: # for RGB image
                        averageTallyRed += I(offX, offY, canvas = RedImgMat)
                        averageTallyGreen += I(offX, offY, canvas = GreenImgMat)
                        averageTallyBlue += I(offX, offY, canvas = BlueImgMat)
                        pixelsAveraged += 1
                    else: # for grayscale image
                        averageTally += I(offX, offY, canvas = imgMat)
                        pixelsAveraged += 1
        if RGB == True:
            if (pixelsAveraged > 0):
                averagesRed.append(int(averageTallyRed/pixelsAveraged))
                averagesGreen.append(int(averageTallyGreen/pixelsAveraged))
                averagesBlue.append(int(averageTallyBlue/pixelsAveraged))
            else:
                averagesRed.append(int((0+255)/2))
                averagesGreen.append(int((0+255)/2))
                averagesBlue.append(int((0+255)/2))
        else:
            if (pixelsAveraged > 0):
                averages.append(int(averageTally/pixelsAveraged))
            else:
                averages.append(int((0+255)/2))
    if RGB == True:
        return [averagesRed, averagesGreen, averagesBlue]
    else:
        return(averages)

''' ~~~~~~~ Setting the mask regions in the image to their average ~~~~~~~~ '''  
def ApplyAveraging(averages):
    print(averages)
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]  
        for y in range(len(submask)):
            for x in range(len(submask[y])):
                if (IsMask(I(x, y, canvas = submask))):
                    offX = x+reducedSubmasksOrigins[i][0]
                    offY = y+reducedSubmasksOrigins[i][1]                    
                    if RGB == True:
                        RedImgMat[offY][offX] = averages[0][i] # red averages
                        GreenImgMat[offY][offX] = averages[1][i] # green averages
                        BlueImgMat[offY][offX] = averages[2][i] # blue averages
                    else:
                        imgMat[offY][offX] = averages[i]

if (isAveraging):
    averages = CalculateAveraging()
    ApplyAveraging(averages)

# test averaging
'''if RGB == True:
    testIm = Image.fromarray(BlueImgMat.astype(np.uint8))
    testImOutPath = "TEST_IMAGE.jpg"
    testIm.save(testImOutPath)
    testIm.show(title = "test") # showing solved image
'''
# Displaying (or save) the reducedSubmasks.
def DisplayOrSaveReducedSubmasks(shouldSave):
    i = 0
    for i in range(0, len(reducedSubmasks)):
        reducedSubmasks[i] = np.array(reducedSubmasks[i])
        reducedSubmaskIm = Image.fromarray(reducedSubmasks[i])
        reducedSubmaskIm = reducedSubmaskIm.convert("L")
        if (shouldSave == False):
            reducedSubmaskIm.show()
        if (shouldSave):
            reducedSubmaskOutPath = "REDUCEDSUBMASK_"+str(i)+".jpg"
            reducedSubmaskIm.save(reducedSubmaskOutPath)
        i+=1

#DisplayOrSaveReducedSubmasks(True)


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Solving ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
startedRunningSolving = time.perf_counter()

if RGB == True:
    workingImgMatRed = deepcopy(RedImgMat)
    workingImgMatGreen = deepcopy(GreenImgMat)
    workingImgMatBlue = deepcopy(BlueImgMat)
else:
    workingImgMat = deepcopy(imgMat)

# separated the methods
def SORgray(workingImgMat, omega = 1, N = 200):
    '''
        SOR method for grayscale image;
        omega = 1 corresponds to Jacobi.
    '''
    workingImgMat = np.array(workingImgMat, dtype = np.int16)
    workingImgMat2 = deepcopy(workingImgMat)
    workingImgMat2 = np.array(workingImgMat2, dtype = np.int16)
    n = 0
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]
        n = 0
        while n < N:   
            for y in range(len(submask)):
                for x in range(len(submask[y])):
                    if (IsMask(I(x, y, canvas = submask))):
                        offX = x+reducedSubmasksOrigins[i][0]
                        offY = y+reducedSubmasksOrigins[i][1]
                        workingImgMat2[offY][offX] = (1-omega) * (workingImgMat[offY][offX]) + (omega/4) * (workingImgMat[offY-1][offX] + workingImgMat[offY+1][offX]\
                                                            + workingImgMat[offY][offX+1] + workingImgMat[offY][offX-1])       
            workingImgMat = deepcopy(workingImgMat2) # Reapplying the Jacobi method onto the primed system
            n+=1
    return(workingImgMat)

def SORRGB(workingImgMatRed, workingImgMatGreen, workingImgMatBlue , omega = 1, N = 200):
    '''
        SOR method for RGB image;
        omega = 1 corresponds to Jacobi.
    '''
    # red
    workingImgMatRed = np.array(workingImgMatRed, dtype = np.int16)
    workingImgMatRed2 = deepcopy(workingImgMatRed)
    workingImgMatRed2 = np.array(workingImgMatRed2, dtype = np.int16)
    # green
    workingImgMatGreen = np.array(workingImgMatGreen, dtype = np.int16)
    workingImgMatGreen2 = deepcopy(workingImgMatGreen)
    workingImgMatGreen2 = np.array(workingImgMatGreen2, dtype = np.int16)
    # blue
    workingImgMatBlue = np.array(workingImgMatBlue, dtype = np.int16)
    workingImgMatBlue2 = deepcopy(workingImgMatBlue)
    workingImgMatBlue2 = np.array(workingImgMatBlue2, dtype = np.int16)
    n = 0
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]
        n = 0
        while n < N:   
            for y in range(len(submask)):
                for x in range(len(submask[y])):
                    if (IsMask(I(x, y, canvas = submask))):
                        offX = x+reducedSubmasksOrigins[i][0]
                        offY = y+reducedSubmasksOrigins[i][1]
                        workingImgMatRed2[offY][offX] = (1-omega) * (workingImgMatRed[offY][offX]) + (omega/4) * (workingImgMatRed[offY-1][offX] + workingImgMatRed[offY+1][offX]\
                                                            + workingImgMatRed[offY][offX+1] + workingImgMatRed[offY][offX-1])
                        workingImgMatGreen2[offY][offX] = (1-omega) * (workingImgMatGreen[offY][offX]) + (omega/4) * (workingImgMatGreen[offY-1][offX] + workingImgMatGreen[offY+1][offX]\
                                                            + workingImgMatGreen[offY][offX+1] + workingImgMatGreen[offY][offX-1]) 
                        workingImgMatBlue2[offY][offX] = (1-omega) * (workingImgMatBlue[offY][offX]) + (omega/4) * (workingImgMatBlue[offY-1][offX] + workingImgMatBlue[offY+1][offX]\
                                                            + workingImgMatBlue[offY][offX+1] + workingImgMatBlue[offY][offX-1]) 
            workingImgMatRed = deepcopy(workingImgMatRed2) # Reapplying the Jacobi method onto the primed system
            workingImgMatGreen = deepcopy(workingImgMatGreen2) # Reapplying the Jacobi method onto the primed system
            workingImgMatBlue = deepcopy(workingImgMatBlue2) # Reapplying the Jacobi method onto the primed system
            n+=1
    return workingImgMatRed, workingImgMatGreen, workingImgMatBlue

if (solverChoice == "SOR" and RGB == True):
    solvedImgMatRed, solvedImgMatGreen, solvedImgMatBlue = SORRGB(workingImgMatRed, workingImgMatGreen, workingImgMatBlue, omega = omega, N = N)
elif (solverChoice == "SOR" and RGB == False):
    solvedImgMat = SORgray(workingImgMat, omega = omega, N = N)



# Manually converting the image back to take values between [0, 254].
for i in range(0, len(reducedSubmasks)):
    submask = reducedSubmasks[i]
    for y in range(len(submask)):
        for x in range(len(submask[y])):
            offX = x+reducedSubmasksOrigins[i][0]
            offY = y+reducedSubmasksOrigins[i][1]
            if RGB == True:
                if solvedImgMatRed[offY][offX] > 254:
                    solvedImgMatRed[offY][offX] = 254
                elif solvedImgMatRed[offY][offX] < 1:
                    solvedImgMatRed[offY][offX] = 0  
                elif solvedImgMatGreen[offY][offX] > 254:
                    solvedImgMatGreen[offY][offX] = 254
                elif solvedImgMatGreen[offY][offX] < 1:
                    solvedImgMatGreen[offY][offX] = 0  
                elif solvedImgMatBlue[offY][offX] > 254:
                    solvedImgMatBlue[offY][offX] = 254
                elif solvedImgMatBlue[offY][offX] < 1:
                    solvedImgMatBlue[offY][offX] = 0  
            else:
                if solvedImgMat[offY][offX] > 254:
                    solvedImgMat[offY][offX] = 254
                elif solvedImgMat[offY][offX] < 1:
                    solvedImgMat[offY][offX] = 0  

''' # Some debugging stuff
outInt = np.zeros((len(reducedSubmasks[0]),len(reducedSubmasks[0][0])))
for i in range(0, len(reducedSubmasks[0])):
    for j in range(0,len(reducedSubmasks[0][i])):
        solvedImMat[i][j] = int(solvedImMat[reducedSubmasksOrigins[0][0]+i][reducedSubmasksOrigins[0][1]+j])
        outInt[i][j] = int(workingImgMat2[reducedSubmasksOrigins[0][0]+i][reducedSubmasksOrigins[0][1]+j])    
   
for i in range(len(outInt)):
    for j in range(len(outInt[0])):
        if outInt[i][j] > 255 or outInt[i][j] < 0 or outInt[i][j] == np.Inf or outInt[i][j] == np.NAN:
            print(outInt[i][j],reducedSubmasksOrigins[0][0]+i,reducedSubmasksOrigins[0][1]+j)
'''              
if RGB == True:
    RGBcomp = [[ [0 for col in range(3)] for col in range(len(RGBImgMat[0]))] for row in range(len(RGBImgMat))]
    for i in range(len(RGBImgMat)):
        for j in range(len(RGBImgMat[0])):
            RGBcomp[i][j][0] = solvedImgMatRed[i][j]
            RGBcomp[i][j][1] = solvedImgMatGreen[i][j]
            RGBcomp[i][j][2] = solvedImgMatBlue[i][j]
    solvedImgMat = np.array(RGBcomp)
    solvedIm = Image.fromarray(solvedImgMat.astype(np.uint8))
    if solvedIm.mode != 'RGB':
        solvedIm = solvedIm.convert('RGB')
else:
    solvedImgMat = np.array(solvedImgMat)
    solvedIm = Image.fromarray(solvedImgMat.astype(np.uint8))
    solvedIm = solvedIm.convert("L")

solvedImOutPath = "SOLVED_IMAGE.jpg"
solvedIm.save(solvedImOutPath)
cpimgIm.show(title = "masked") #showing masked image
solvedIm.show(title = "solved") # showing solved image
'''-------------------------------------------------------'''


''' ~~~~~~~~~~~~~~~~~~~~~~~~~ Discrepancy scoring ~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
    Comparing the graffiti sprayed regions to those of those regions in the original image.
'''
startedRunningDiscrepancy = time.perf_counter()

def CalculateDiscrepancyScore():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discrepancy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in range(0, len(reducedSubmasks)):
        diffsq=0
        sigmasq=0
        origIsum=0
        submask = reducedSubmasks[i]
        xOff = reducedSubmasksOrigins[i][0]
        yOff = reducedSubmasksOrigins[i][1]
        subYlength = len(submask)
        # get average intensity 'Imean' in submask area on original image
        if len(reducedSubmasks) > 0:
            for y in range(subYlength):
                subXlength = len(submask[y])
                subNumPix = subYlength*subXlength
                for x in range(subXlength):
                    offX = x+xOff
                    offY = y+yOff
                    origIsum = origIsum + int(I(offX, offY, canvas = oimgMat))
                    
            Imean = origIsum/subNumPix            
            # get chi squared
            for y in range(subYlength):
                for x in range(len(submask[y])):
                    offX = x+xOff
                    offY = y+yOff            
                    diffsq = diffsq + ((int(I(offX, offY, canvas = solvedImgMat))-int(I(offX, offY, canvas = oimgMat)))**2)
                    sigmasq = sigmasq + int((I(offX, offY, canvas = oimgMat) - Imean)**2)                                
            if subNumPix > 0 and sigmasq > 0:                           
                sigmasq = sigmasq/(subNumPix-1)            
                print('SubMask number:', i+1, '\u03C7\u00B2 =',(diffsq/subNumPix)/sigmasq)
            else:
                print('SubMask number:', i+1, '\u03C7\u00B2 =',"NaN")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()   
        
CalculateDiscrepancyScore()
    
    
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End timer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
finishedRunning = time.perf_counter()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User Choices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Preparation duration: " + str(startedRunningSolving - startedRunning) + " s.")
print("Jacobi duration: " + str(startedRunningDiscrepancy - startedRunningSolving) + " s.")
print("Discrepancy duration: " + str(finishedRunning - startedRunningDiscrepancy) + " s.")
print("Total duration: " + str(finishedRunning - startedRunning) + " s.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print()