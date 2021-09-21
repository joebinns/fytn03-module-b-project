

''' ~~~~~~~~~~~~~~~~~~~~~~~~~ Check if greyscale ~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
    Since an image that appears to be grayscale may actually be storing some
    colour values (i.e. a digital colour photograph taken of a greyscale image
    may interpret some colour, such as a general hue). I will check that the image
    is truly grayscale here, and if it is found not to be, I will apply the 
    greyscale filter to the image in an image editing software, and reimport.
    
    Or... actually the next section of code converts image to greyscale anyway when importing...
'''
from PIL import Image

def IsGreyScale(img_path): # Source: https://stackoverflow.com/a/23661373
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

#print(IsGreyScale("LOTF_RGB.jpg")) # Image Source: https://www.moma.org/explore/inside_out/2013/11/12/peter-brooks-lord-of-the-flies/
#print(IsGreyScale("LOTF_GS.jpg"))

#imgInPath = "LOTF_GS_FLIED.jpg"
imgInPath = "LOTF_GS_RECTANGLED.jpg"


''' ~~~~~~~~~~~~~~~~~~ Import image and convert to matrix ~~~~~~~~~~~~~~~~~ '''
'''
    Note: Divide the matrix by 255 to convert to intesities ranging from 0 to 1.
'''
import numpy as np

imgMat = np.array(Image.open(imgInPath).convert('L')) # Converts to greyscale matrix!
#print(imgMat)

def I(x,y,mat = imgMat):
    '''
    Input(s):
    Note: Origin of the image is treated as the top-left-most pixel, and takes value [0,0]!
    x: x coordinate of the pixel, in the range [0, imageWidth - 1].
    y: y coordinate of hte pixel, in the range [0, imageHeight - 1].
    mat: the matrix of the image.
    
    Output(s):
    The intensity of the pixel at coordinate (x, y), taking values [0, 255] corresponding to [black, white].
    '''
    #matHeight = len(mat)
    #matWidth = len(mat[0])
    
    row = y
    column = x
    
    return(mat[row][column])

''' ~~~~~~~~~~~~~~~ Checking image is interpreted correctly ~~~~~~~~~~~~~~~ '''
imgOutPath = "LOTF_OUT.png"
Image.fromarray(imgMat).save(imgOutPath) # Exports image!



''' ~~~~~~~~~~~~~~~~~~ Import mask and convert to matrix ~~~~~~~~~~~~~~~~~~ '''
#maskInPath = "LOTF_GS_FLIES.jpg"
maskInPath = "LOTF_GS_RECTANGLES.jpg"

maskMat = np.array(Image.open(maskInPath).convert('L')) # Converts transparency in png to whiteness, and retains blackness.
#print(maskMat)



''' ~~~~~~~~~~~~~~~ Checking mask is interpreted correctly ~~~~~~~~~~~~~~~~ '''
maskOutPath = "MASK_OUT.png"
Image.fromarray(maskMat).save(maskOutPath) # Exports mask!
        


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Recover image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
    Using Harmonic inpainting:
        --> Numerically solving the Laplace equation for a missing region using the intensities at the edges as boundary values.
        
    Solutions to Laplace's equations are called potential fields.
'''
import numpy as np


# TO DO: GOOGLE HARMONIC INPAINTING, CHECK LECTURE SLIDES / NOTES / WATCH LECTURES.

def SquareNormError(p, pn):
    '''
        Input(s):
        p: Step n+1's estiamte of p.
        pn: Step n's estimate of p.
        
        Output(s):
        output: Normalised squared difference between succesive estimates,
                taken across the entire grid.
    '''
    sN = np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))
    return(sN)

def Jacobi(p, sNTarget):
    '''
        Iteratively solves the Laplace equation using the Jacobi method.

        Input(s):
        p [2D array of float]: Initial potential distribution.
        sNTarget [float]: target for the difference between consecutive solutions.

        Ouput(s):
        p [2D array of float]: Potential distribution after relaxation.
            
    '''
    sNNorm = 1
    pn = np.empty_like(p)
    while (sNNorm > sNTarget):
        pn = p.copy()
        p[1:-1,1:-1] = (1/4) * (pn[1:-1,2:] + pn[1:-1, :-2] \
                              + pn[2:, 1:-1] + pn[:-2, 1:-1])

        ##Neumann B.C. along x = L
        p[1:-1, -1] = p[1:-1, -2]     # 1st order approx of a derivative 
        sNNorm = SquareNormError(p, pn)
       
    return (p)

 
##variable declarations
imgHeight = len(imgMat)
imgWidth = len(imgMat[0])    

nx = imgWidth
ny = imgHeight


##initial conditions
p = np.zeros((ny,nx)) ##create a XxY vector of 0's


##plotting aids
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)

##Dirichlet boundary conditions
p[-1,:] = np.sin(1.5*np.pi*x/x[-1])  



''' 3D PLOTTING... '''
"""
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  

def plot_3D(x, y, p):
    '''Creates 3D plot with appropriate limits and viewing angle

    Parameters:
    ----------
    x: array of float
        nodal coordinates in x
    y: array of float
        nodal coordinates in y
    p: 2D array of float
        calculated potential field

    '''
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(30,45)

p = Jacobi(p.copy(), 0.01)
plot_3D(x, y, p)
"""

''' ~~~~~~~~~~~~~~~~~~~~~~~~~ Discrepancy scoring ~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
    Comparing the graffiti sprayed regions to those of those regions in the original image.
'''