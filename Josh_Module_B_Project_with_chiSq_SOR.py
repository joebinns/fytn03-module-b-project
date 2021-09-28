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

def ValidatedInput(inputQueue, answerRange):
    while True: # Check that the input is a number, and then check that the number is within the desired range for answers.
        try:
            theInput = int(input(inputQueue))
            while True: # Check that the input, which has already been established to be a number, is within the desired range for answers.
                if (theInput >= answerRange[0] and theInput <= answerRange[1]):
                    break # Escape the while loop if the input is within the range.
                else:
                    print("Input was not within the range of the options. Please re-enter.")
                    theInput = int(input(inputQueue))                    
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

# Choose oimage!
oimageFilePaths = ["LOTF_GS.jpg", "LOTF_RGB.jpg"]
oimagePathChoice = oimageFilePaths[QuestionAndAnswer("Select the file path of the original image.", oimageFilePaths)]
print()

# Choose mask (and therefore also the masked image)
maskFilePathsAll = {
        "LOTF_GS.jpg": ["LOTF_GS_RECTANGLES.jpg", "LOTF_GS_FLIES.jpg", "LOTF_GS_BIGFLIES.jpg", "LOTF_GS_TEXT.jpg"],
        "LOTF_RGB.jpg": ["LOTF_RGB_RECTANGLES.jpg", "LOTF_RGB_FLIES.jpg", "LOTF_RGB_BIGFLIES.jpg", "LOTF_RGB_TEXT.jpg"]
        }
maskFilePaths = maskFilePathsAll[oimagePathChoice]
maskPathChoice = maskFilePaths[QuestionAndAnswer("Select the file path of the mask.", maskFilePaths)]
print()

# Automatically determine the masked version of the image.
imageFilePathsGS = {
        "LOTF_GS_RECTANGLES.jpg": "LOTF_GS_RECTANGLED.jpg",
        "LOTF_GS_FLIES.jpg": "LOTF_GS_FLIED.jpg",
        "LOTF_GS_BIGFLIES.jpg": "LOTF_GS_BIGFLIED.jpg",
        "LOTF_GS_TEXT.jpg": "LOTF_GS_TEXTED.jpg"
        }
imageFilePathsRGB = {
        "LOTF_RGB_RECTANGLES.jpg": "LOTF_RGB_RECTANGLED.jpg",
        "LOTF_RGB_FLIES.jpg": "LOTF_RGB_FLIED.jpg",
        "LOTF_RGB_BIGFLIES.jpg": "LOTF_RGB_BIGFLIED.jpg",
        "LOTF_RGB_TEXT.jpg": "LOTF_RGB_TEXTED.jpg"
        }
imageFilePathsAll = {
        "LOTF_GS.jpg": imageFilePathsGS,
        "LOTF_RGB.jpg": imageFilePathsRGB
        }

imagePathChoice = imageFilePathsAll[oimagePathChoice][maskPathChoice]
print()

isAveragingOptions = ["True", "False"]
isAveraging = isAveragingOptions[QuestionAndAnswer("Apply colour-averaging to the mask using it's surroundings, prior to applying the solving method?", isAveragingOptions)]
if (isAveraging == "True"):
    isAveraging = True
else:
    isAveraging = False

""" WORK IN PROGRESS
# Choose solver!
solvers = ["SOR"]
solverChoice = solvers[QuestionAndAnswer("Select the solver to be used!", solvers)]
print()

SOROptions = {"Number of iterations (N):": [1,200],
              "Relaxation (omega):": [0.01, 1.99]
              }
solversAll = {
        "SOR": SOROptions
        }

for key in solversAll[solverChoice]:
    allowedRange = solversAll[solverChoice][key]
    
    QuestionAndAnswer()
"""    

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User Choices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("oimagePathChoice:", oimagePathChoice)
print("imagePathChoice:", imagePathChoice)
print("maskPathChoice:", maskPathChoice)
print()
print("isAveraging:", str(isAveraging))
#print("solverChoice:", solverChoice) WORK IN PROGRESS
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print()


''' ~~~~~~~~~~~~~~~~~ Import original image and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
oimgInPath = oimagePathChoice # path of the original, unmasked image.
oimgIm = Image.open(oimgInPath)
oimgIm = oimgIm.convert('L') # converts image to grayscale.
oimgMat = np.array(oimgIm, dtype = np.int16) # gets array of image.


''' ~~~~~~~~~~~~~~~~~ Import image and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
imgInPath = imagePathChoice # path of the image with mask applied.
imgIm = Image.open(imgInPath)
imgIm = imgIm.convert('L') # converts image to grayscale.
imgMat = np.array(imgIm, dtype = np.int16) # gets array of image.

#canvasWidth, canvasHeight = imgIm.size #defines the matrix dims 


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

''' ~~~~~~~~~~~ Calculating the average from all the boundaries ~~~~~~~~~~~ '''
def CalculateAveraging():
    averages = []
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]
        averageTally = 0
        pixelsAveraged = 0   
        for y in range(len(submask)):
            for x in range(len(submask[y])):
                if (IsMask(I(x, y, canvas = submask)) == False):
                    offX = x+reducedSubmasksOrigins[i][0]
                    offY = y+reducedSubmasksOrigins[i][1]
                    averageTally += I(offX, offY, canvas = imgMat)
                    pixelsAveraged += 1
                    if (pixelsAveraged > 0):
                        averages.append(int(averageTally/pixelsAveraged))
                    else:
                        averages.append(int((0+255)/2))
    return(averages)

''' ~~~~~~~ Setting the mask regions in the image to their average ~~~~~~~~ '''  
def ApplyAveraging(averages):
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]  
        for y in range(len(submask)):
            for x in range(len(submask[y])):
                if (IsMask(I(x, y, canvas = submask))):
                    offX = x+reducedSubmasksOrigins[i][0]
                    offY = y+reducedSubmasksOrigins[i][1]                    
                    imgMat[offY][offX] = averages[i]

if (isAveraging):
    averages = CalculateAveraging()
    ApplyAveraging(averages)

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

workingImgMat = deepcopy(imgMat)

def SOR(workingImgMat, omega = 1, N = 200):
    '''
        omega = 1 corresponds to Jacobi.
    '''
    workingImgMat = np.array(workingImgMat, dtype = np.int16)
    workingImgMat2 = deepcopy(workingImgMat)
    workingImgMat2 = np.array(workingImgMat2, dtype = np.int16)
    N =  200 # 1000 #count of loops for Jacobi method
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
                        workingImgMat2[offY][offX] = workingImgMat2[offY][offX] = (1-omega) * (workingImgMat[offY][offX])\
                                                        + (omega/4) * (workingImgMat[offY-1][offX] + workingImgMat[offY+1][offX]\
                                                            + workingImgMat[offY][offX+1] + workingImgMat[offY][offX-1])       
            workingImgMat = deepcopy(workingImgMat2) # Reapplying the Jacobi method onto the primed system
            n+=1
    return(workingImgMat)

if (solverChoice == "SOR"):
    solvedImgMat = SOR(workingImgMat, omega = 1, N = 200)

# Manually converting the image back to take values between [0, 254].
for i in range(0, len(reducedSubmasks)):
    submask = reducedSubmasks[i]
    for y in range(len(submask)):
        for x in range(len(submask[y])):
            offX = x+reducedSubmasksOrigins[i][0]
            offY = y+reducedSubmasksOrigins[i][1]
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
        
solvedImgMat = np.array(solvedImgMat, dtype = np.int8) 

solvedImOutPath = "SOLVED_IMAGE.jpg"
solvedIm = Image.fromarray(solvedImgMat)
solvedIm = solvedIm.convert("L")
solvedIm.save(solvedImOutPath)

imgIm.show(title = "masked") #showing masked image
solvedIm.show(title = "solved") # showing solved image


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
                    diffsq = diffsq + ((int(I(offX, offY, canvas = solvedImMat))-int(I(offX, offY, canvas = oimgMat)))**2)
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

