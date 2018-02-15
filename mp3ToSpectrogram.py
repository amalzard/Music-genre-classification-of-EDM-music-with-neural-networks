import pydub
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3
import sys
import datetime

#directories for file I/O
currentPath = os.path.dirname(os.path.realpath(__file__))
mp3Folder=currentPath+"/mp3/"
wavFolder=currentPath+"/wav/"
spectrogramsPath=currentPath+"/spectrograms/"


#Extracts the genre from an mp3 file. Returns None if field is blank
def getGenre(filename):
	#print(filename)
	audiofile = eyed3.load(filename)
	if not audiofile.tag.genre:
		return None
	else:
		return audiofile.tag.genre.name.encode('utf-8')

#Converts the mp3 file to .wav, downmixes to mono and then creates a spectrogram of the .wav and deletes the .wav
def createSpectrogram(filename, newFilename, fileGenre):
	mp3 = pydub.AudioSegment.from_mp3(mp3Folder+filename)
	#mix stereo to mono
	mp3 = mp3.set_channels(1)
	#convert to wav
	mp3.export(wavFolder+newFilename, format="wav")
	exportedFile = wavFolder+newFilename
	pixelPerSecond = 50
	command = "sox '{}' -n spectrogram -y 128 -X {} -h -r -o '{}.png'".format(exportedFile,pixelPerSecond,spectrogramsPath+"/"+newFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .wav...")
	os.remove(exportedFile)

#coverts flac to mp3, keeping tags. Then deletes the flac
def convertFlacToMp3(filename):
	convertedFilename = str(datetime.datetime.now()).split('.')[0]
	command = 'ffmpeg -loglevel panic -i "{}" -ab 320k -map_metadata 0 -id3v2_version 3 "{}.mp3"'.format(mp3Folder+filename, mp3Folder+convertedFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .flac...")
	os.remove(mp3Folder+filename)


def convertOogToMp3(filename):
	convertedFilename = str(datetime.datetime.now()).split('.')[0]
	command = 'ffmpeg -loglevel panic -i "{}" -ab 320k -map_metadata 0:s:0 -acodec libmp3lame "{}.mp3"'.format(mp3Folder+filename, mp3Folder+convertedFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors
	print("Deleting .ogg...")
	os.remove(mp3Folder+filename)

def renameMp3Files():
	musicFiles = os.listdir(mp3Folder)
	mp3Files = [file for file in musicFiles if file.endswith(".mp3")]
	nbFiles = len(mp3Files)
	if len(mp3Files) > 0:
		for index, filename in enumerate(mp3Files):
			print "Renaming file {}/{}...".format(index+1,nbFiles)
			convertedFilename = (str(datetime.datetime.now()).split('.')[0] + "_" + str(index) + ".mp3")
			os.rename(mp3Folder+filename, mp3Folder+convertedFilename)


genresID = dict()
musicFiles = os.listdir(mp3Folder)
flacFiles = [file for file in musicFiles if file.endswith(".flac")]
nbFiles = len(flacFiles)
if len(flacFiles) > 0:
	print("Found flac files. Converting to mp3...")
	for index, flacFilename in enumerate(flacFiles):
		print "Converting file {}/{}...".format(index+1,nbFiles)
		convertFlacToMp3(flacFilename)

musicFiles = os.listdir(mp3Folder)
oggFiles = [file for file in musicFiles if file.endswith(".ogg")]
nbFiles = len(oggFiles)
if len(oggFiles) > 0:
	print("Found ogg files. Converting to mp3...")
	for index, oggFilename in enumerate(oggFiles):
		print "Converting file {}/{}...".format(index+1,nbFiles)
		convertOogToMp3(oggFilename)

renameMp3Files()
musicFiles = os.listdir(mp3Folder)
mp3Files = [file for file in musicFiles if file.endswith(".mp3")]
nbFiles = len(mp3Files)
if len(mp3Files) > 0:
	for index, filename in enumerate(mp3Files):
		print "Creating spectrogram for file {}/{}...".format(index+1,nbFiles)
		fileGenre = getGenre(mp3Folder+filename)
		genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
		fileID = genresID[fileGenre]
		if fileGenre == None:
			fileGenre = "noGenre"
		newFilename = fileGenre+"_"+str(fileID)
		createSpectrogram(filename, newFilename, fileGenre)
	print("Spectrograms Created!")
else:
	print("No mp3 files in mp3 folder")

