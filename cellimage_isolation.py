#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:31:59 2025

@author: jankae
"""

import numpy as np
from PIL import Image


#Opening the cell-mask and the correspondingiamge
im_mask = Image.open('/home/jankae/Downloads/5_masks(1).tif')
im_real = Image.open('/home/jankae/Blood_samples/Blood/blood-SH/5.jpg')

#Setting up the array containing the cell coordinates
cellsbase = np.array((im_mask.getdata()))
cells = cellsbase[:]
cells_final = cells.reshape(3286,4096)

#Initializing the variables for the loops and the cropping of the individual cell images
coordslistX = []
coordslistY = []
cellnumber = 1
i= 1
y= 0
x=0
xcoord = 0
ycoord = 0
cropcordx = 0
cropcordy = 0
whiteoutx = []
whiteouty= []
y_min =0
y_max =0
x_min=0
x_max=0
j=0
p=0
l=1
RGB= (255,255,255)


#looping through the desired mmount of cells, number behind i can be adjusted as seen fit
while i < 180:

    while y < 3285:
    
        while x < 4095 :
        
            if cells_final[y, x] == cellnumber:
#listing the coordinates of the cell by tracking the position within the array for all entries of the given cellnumber
                coordslistX.append(x)
                coordslistY.append(y)
                
            x= x+1
        y = y+1
        x =0
    f=1
    t=1
    y=0
    

#Determening the rough center of the cell by calculating the mean of both the X and Y coordinates
    if len(coordslistX) != 0 and len(coordslistY) !=0:
        cropcordx = int(np.mean(coordslistX))
        cropcordy = int(np.mean(coordslistY))


#Using the determined cell center to crop out a 256 by 256 pixel image
        heightup = cropcordy+128
        if heightup > 3286:
            heightup = 3286

        heightdown = cropcordy-128
        if heightdown < 0:
            heightdown = 0
            
        widthleft = cropcordx-128
        if widthleft < 0:
            widthleft = 0
   
        widthright = cropcordx+128
        if widthright > 4096:
            widthright = 4096
        cropspace= (widthleft, heightdown, widthright, heightup)
        cropim= im_real.crop(cropspace)
        cropdata = cropim.load()

#temporarily saving the cropped image to perform the whiting out of everything except the current cell
        filestringproto= '/home/jankae/Blood_samples/temp1/temp'+str(i)
        filestring= filestringproto+'.tiff'
        cropim.save(filestring)
        cropped_im= Image.open(filestring)
        
#Initializing Variables for the whiting out
        h = heightdown
        w = widthleft
        o=0
        g=0
        
#Looping through the pixels of the cropped image and whiting out every pixel with values differing from the current cell number in the cells_final array
        while h < heightup:
            while w < widthright:
                
                if cells_final[h,w] != cellnumber:
                    cropped_im.putpixel((0+g,0+o),(255,255,255) )
                w =w+1
                if g+1 <= widthright-widthleft:
                    g=g+1
            h=h+1
            if o+1 <= heightup-heightdown:
                o=o+1
            g=0
            w=widthleft
            
#Saving the final image
        filestringfinal= '/home/jankae/Blood_samples/whited_singles_SH/cell'+str(l)
        filestring= filestringfinal+'.tiff'
        cropped_im.save(filestring)
        l=l+1
        
#Resetting and increasing loop variables to move onto the next cell        
    cellnumber = cellnumber + 1
    coordslistX = []
    coordslistY = []
    xcoord = 0
    ycoord = 0
    p=0
    j=0
    i = i+1