# -*- coding: utf-8 -*-
import random
import string
import os
import sys
import numpy as np
from collections import Counter

from model import createModel
from dataset import getDataset
from config import batchSize
from config import filesPerGenre
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceSize
from config import genre_dict
from processPedictionFiles import createPredictionSpectrogram, convertPredictionFlacToMp3, convertPredictionOogToMp3, renamePredictionMp3Files, convertPredictionMp3ToSpectrogram
from slicePredictionFiles import splitPredictionSpectrogram, slicePredictionSpectrograms

currentPath = os.path.dirname(os.path.realpath(__file__))
mp3Folder=currentPath+"/mp3/"
spectrogramsPath=currentPath+"/spectrograms/"
slicesPath=currentPath+"/spectrogramSlices/"
predictPath=currentPath+"/filesToPredict/"

from mp3ToSpectrogram import convertMp3ToSpectrogram
from splitSpectrograms import sliceSpectrograms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","convert","slice","predict"])
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Slices per genre: {}".format(filesPerGenre))
print("| Slice size: {}".format(sliceSize))
print("--------------------------")

if "convert" in args.mode:
	convertMp3ToSpectrogram()
	sys.exit()

if "slice" in args.mode:
	sliceSpectrograms()
	sys.exit()

#List genres
genres = os.listdir(slicesPath)
genres = [filename for filename in genres if os.path.isdir(slicesPath+filename)]
nbClasses = len(genres)
#print(str(genres))

#Create model 
model = createModel(nbClasses, sliceSize)

if "train" in args.mode:

	#Create or load new dataset
	train_X, train_y, validation_X, validation_y, genre_dict = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, genre_dict, mode="train")

	#Define run id for graphs
	run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

	#Train the model
	print("[+] Training the model...")
	model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
	print("    Model trained! ✅")

	#Save trained model
	print("[+] Saving the weights...")
	model.save('musicDNN.tflearn')
	print("[+] Weights saved! ✅💾")

if "test" in args.mode:

	#Create or load new dataset
	test_X, test_y, genre_dict = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, genre_dict, mode="test")

	#Load weights
	print("[+] Loading weights...")
	model.load('musicDNN.tflearn')
	print("    Weights loaded! ✅")

	testAccuracy = model.evaluate(test_X, test_y)[0]
	print("[+] Test accuracy: {} ".format(testAccuracy))

if "predict" in args.mode:

	#Create or load new dataset
	test_X, test_y, genre_dict = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, genre_dict, mode="test")

	#Load weights
	print("[+] Loading weights...")
	model.load('musicDNN.tflearn')
	print("    Weights loaded! ✅")
	convertPredictionMp3ToSpectrogram()
	slicePredictionSpectrograms()
	sliceList = []
	predictionList = []
	from PIL import Image, ImageOps
	#prediction = model.predict([test_X[0]])
	#predictionLabel = model.predict_label([test_X[1]])
	#label_name = [genres[id] for id in predictionLabel[0]]
	#splitPredictions = np.split(predictionLabel[0], nbClasses)
	#splitProbabilities = np.split(prediction[0], nbClasses)
	#predictInt = int(splitPredictions[0])
	#for key, value in genre_dict.items():
		#if predictInt == value:
			#predictionClass = key
	#print("Prediction: " + predictionClass + " Probability: " + str(splitProbabilities[0]))
	fileList = os.listdir(predictPath)
	sliceFiles = [file for file in fileList if file.endswith(".png")]
	nbFiles = len(sliceFiles)
	if len(sliceFiles) > 0:
		for i in range(len(sliceFiles)):
			
			img = Image.open(predictPath+"file_0_"+ str(i) +".png")
			img = ImageOps.fit(img, ((128,128)), Image.ANTIALIAS)

			img_arr = np.array(img)
			img_arr = img_arr.reshape(-1,128,128,1).astype("float")
			print("Adding " + "file_0_"+ str(i) +".png" + " to sliced image list...")

			#sliceList.append(img_arr)
			pred = model.predict(img_arr)

			predictInt = np.argmax(pred)
			for key, value in genre_dict.items():
				if predictInt == value:
					predictionClass = key

					predictionList.append(predictionClass)

	predictionCounter = (prediction for prediction in predictionList)

	c = Counter(predictionCounter)
	print c.most_common(3)





	



	#img = Image.open(predictPath+"test.png")
	#img = ImageOps.fit(img, ((128,128)), Image.ANTIALIAS)

	#img_arr = np.array(img)
	#img_arr = img_arr.reshape(-1,128,128,1).astype("float")

	#pred = model.predict(img_arr)
	#predLabel = model.predict_label(img_arr)

	#splitPredictions = np.split(predLabel[0], nbClasses)
	#splitProbabilities = np.split(pred[0], nbClasses)

	


	#print("Results: %s"%predictionClass)
	#print(predictionClass + " " + str(model.predict(img_arr)))



