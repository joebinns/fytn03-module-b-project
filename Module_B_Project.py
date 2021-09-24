from PIL import Image
import numpy as np

''' ~~~~~~~~~~~~~~~~~ Import image and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
imgInPath = "LOTF_GS_RECTANGLED.jpg" # path of the image with mask applied
#imgInPath = "LOTF_GS_FLIED.jpg"
imgIm = Image.open(imgInPath)
imgIm = imgIm.convert('L') # converts image to grayscale.
imgMat = np.array(imgIm) # gets array of image.

width, length = imgIm.size #defines the matrix dims 

''' ~~~~~~~~~~~~~~~~~~ Import mask and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
maskInPath = "LOTF_GS_RECTANGLES.jpg" # path of the mask which has been applied
#maskInPath = "LOTF_GS_FLIES.jpg"
maskIm = Image.open(maskInPath)
maskIm = maskIm.convert('L') # converts mask image to grayscale.
maskMat = np.array(maskIm) # gets array of mask image.

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
    if (i == 0 or i == 1): # i == 1 is a temporary fix. For some unknown reason, the imported mask matrix has values of 0 and 1 (on a scale of [0, 255]) for the black, whereas we expect only 0. 
        isMask = True
    return isMask

def FindNeighbours(canvas, submasks, p):
    '''   
    Input(s):
        canvas {nested array}: The matrix of the mask. 
        submasks {array of nested arrays}: List of all the submasks matrices. 
        p {length-2 array}: Coordinate of the pixel to work from in the form [x, y].
    Output(s):
        neighbours {array of length-2 arrays}: A collection of all the appropriate neighbours (i.e. within bounds of canvas, not already contained in another submask, has intensity expected of mask).
    '''
    possibleNeighbours = [
        [p[0], p[1] + 1], # South
        [p[0] + 1, p[1]], # East
        [p[0], p[1] - 1], # West
        [p[0] - 1, p[1]] # North
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
        neighbours = FindNeighbours(canvas, submasks, currentLoc)
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
reducedSubmasks = []
reducedSubmasksOrigins = []
for submask in submasks:
    reducedSubmask = []  
    minX = len(imgMat[0])
    minY = len(imgMat)
    maxX = 0
    maxY = 0
    for y in range(0, len(imgMat)):
        for x in range(0, len(imgMat[0])):
            if (IsMask(I(x, y, canvas = submask))):
                if (x < minX):
                    minX = x
                elif (x > maxX):
                    maxX = x
                if (y < minY):
                    minY = y
                elif (y > maxY):
                    maxY = y
    for y in range(minY, maxY+1):
        row = []
        for x in range(minX, maxX+1):
            row.append(I(x, y, canvas = submask))
        reducedSubmask.append(row)
    reducedSubmasks.append(reducedSubmask)
    reducedSubmasksOrigins.append([minX, minY])

for submask in reducedSubmasks:
    print(submask)
    
print(len(reducedSubmasks))

''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Solving ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

#wmaskpri = wmask #defining my working image for Jacobi method
wmask = imgMat
wmaskpri = imgMat

N = 1000 #count of loops for Jacobi method
for e in range(N):   
    for i in range(0, len(reducedSubmasks)):
        submask = reducedSubmasks[i]
        xOff = reducedSubmasksOrigins[i][0]
        yOff = reducedSubmasksOrigins[i][1]
        for y in range(len(submask)):
            for x in range(len(submask[y])):
                if (IsMask(I(x, y, canvas = submask))):
                    offX = x+xOff
                    offY = y+yOff
                    wmaskpri[offY,offX] = (int(wmask[offY-1,offX]) + int(wmask[offY+1,offX]) + int(wmask[offY,offX+1]) + int(wmask[offY,offX-1]))/4
    wmaskpri, wmask = wmask, wmaskpri #reapplying the Jacobi method onto the primed system

solvedImOutPath = "LOTF_GS_DERECTANGLED.jpg"
solvedIm = Image.fromarray(wmaskpri)
solvedIm = solvedIm.convert("L")
solvedIm.save(solvedImOutPath)

imgIm.show(title = "masked") #showing masked image
solvedIm.show(title = "solved") # showing solved image


''' ~~~~~~~~~~~~~~~~~~~~~~~~~ Discrepancy scoring ~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
    Comparing the graffiti sprayed regions to those of those regions in the original image.
'''