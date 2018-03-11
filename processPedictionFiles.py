import pydub
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3
import sys
import datetime
from config import pixelPerSecond

#directories for file I/O
currentPath = os.path.dirname(os.path.realpath(__file__))
mp3Folder=currentPath+"/mp3/"
wavFolder=currentPath+"/wav/"
spectrogramsPath=currentPath+"/spectrograms/"
predictionPath=currentPath+"/filesToPredict/"


#Converts the mp3 file to .wav, downmixes to mono and then creates a spectrogram of the .wav and deletes the .wav
def createPredictionSpectrogram(filename):
	mp3 = pydub.AudioSegment.from_mp3(predictionPath+filename)
	#mix stereo to mono
	mp3 = mp3.set_channels(1)
	#convert to wav
	mp3.export(wavFolder+filename, format="wav")
	exportedFile = wavFolder+filename
	command = "sox '{}' -n spectrogram -Y 200 -X {} -m -z 80 -r -o '{}.png'".format(exportedFile,pixelPerSecond,predictionPath+"/"+filename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .wav...")
	os.remove(exportedFile)

#coverts flac to mp3, keeping tags. Then deletes the flac
def convertPredictionFlacToMp3(filename):
	convertedFilename = str(datetime.datetime.now()).split('.')[0]
	command = 'ffmpeg -loglevel panic -i "{}" -ab 320k -map_metadata 0 -id3v2_version 3 "{}.mp3"'.format(predictionPath+filename, predictionPath+convertedFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .flac...")
	os.remove(predictionPath+filename)


def convertPredictionOogToMp3(filename):
	convertedFilename = str(datetime.datetime.now()).split('.')[0]
	command = 'ffmpeg -loglevel panic -i "{}" -ab 320k -map_metadata 0:s:0 -acodec libmp3lame "{}.mp3"'.format(predictionPath+filename, predictionPath+convertedFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .ogg...")
	os.remove(predictionPath+filename)

def renamePredictionMp3Files():
	musicFiles = os.listdir(predictionPath)
	mp3Files = [file for file in musicFiles if file.endswith(".mp3")]
	nbFiles = len(mp3Files)
	if len(mp3Files) > 0:
		for index, filename in enumerate(mp3Files):
			print "Renaming file {}/{}...".format(index+1,nbFiles)
			convertedFilename = (str(datetime.datetime.now()).split('.')[0] + "_" + str(index) + ".mp3")
			os.rename(predictionPath+filename, predictionPath+convertedFilename)


def convertPredictionMp3ToSpectrogram():
	musicFiles = os.listdir(predictionPath)
	flacFiles = [file for file in musicFiles if file.endswith(".flac")]
	nbFiles = len(flacFiles)
	if len(flacFiles) > 0:
		print("Found flac files. Converting to mp3...")
		for index, flacFilename in enumerate(flacFiles):
			print "Converting file {}/{}...".format(index+1,nbFiles)
			convertPredictionFlacToMp3(flacFilename)

	musicFiles = os.listdir(predictionPath)
	oggFiles = [file for file in musicFiles if file.endswith(".ogg")]
	nbFiles = len(oggFiles)
	if len(oggFiles) > 0:
		print("Found ogg files. Converting to mp3...")
		for index, oggFilename in enumerate(oggFiles):
			print "Converting file {}/{}...".format(index+1,nbFiles)
			convertPredictionOogToMp3(oggFilename)

	renamePredictionMp3Files()
	musicFiles = os.listdir(predictionPath)
	mp3Files = [file for file in musicFiles if file.endswith(".mp3")]
	nbFiles = len(mp3Files)
	if len(mp3Files) > 0:
		for index, filename in enumerate(mp3Files):
			print "Creating spectrogram for file {}/{}...".format(index+1,nbFiles)
			createPredictionSpectrogram(filename)
		print("Prediction Spectrograms Created!")
	else:
		print("No mp3 files in filesToPredict folder")

