import os
import sys
from PIL import Image


currentPath = os.path.dirname(os.path.realpath(__file__))
mp3Folder=currentPath+"/mp3/"
spectrogramsPath=currentPath+"/spectrograms/"
slicesPath=currentPath+"/spectrogramSlices/"

def splitSpectrogram(filename):
	print(filename)
	img = Image.open(spectrogramsPath + filename)
	genre = filename.split("_")[0]
	width, height = img.size
	nbSamples = int(width / 128)
	sliceFilename = filename.split(".")[0]
	slicePath = slicesPath+"{}/".format(genre);
	if not os.path.exists(os.path.dirname(slicePath)):
		os.makedirs(os.path.dirname(slicePath))
	for i in range(nbSamples):
		print "Creating slice: ", (i+1), "/", nbSamples, "for", filename
		startPixel = i*128
		imgTmp = img.crop((startPixel, 1, startPixel + 128, 128 + 1))
		imgTmp.save(slicesPath+"{}/{}_{}.png".format(genre,sliceFilename,i))




def sliceSpectrograms():

	spectrogramFiles = os.listdir(spectrogramsPath)
	pngFiles = [file for file in spectrogramFiles if file.endswith(".png")]
	nbFiles = len(spectrogramFiles)
	if len(spectrogramFiles) > 0:
		for index, filename in enumerate(spectrogramFiles):
			splitSpectrogram(filename)
		print("Slices Created!")
	else:
		print("No spectrograms to slice")