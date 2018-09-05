import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils2 import *
from constants import *
from models.model_bce import ModelBCE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
VideoNameFile = 'Testlist.txt'	# choose the data
Video_dir = '/home/s/re/LEDOV-down'
w = 256 
h = 144

def main():
	# Create network
	#model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=1)
	model = ModelBCE(448, 448, batch_size=1)
	# Here need to specify the epoch of model sanpshot
	load_weights(model.net['output'], path='gen_', epochtoload=145)
	# Here need to specify the path to images and output path
	videofile = open(VideoNameFile, 'r')
	allline = videofile.readlines()
	for line in allline:
		lindex = line.index('\t')
		VideoIndex = int(line[:lindex])
		VideoName = line[lindex + 1:-2]
		print VideoName
		VideoCap = cv2.VideoCapture(Video_dir + '/' + VideoName + '.mp4')
		fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
		VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
		videoWriter = cv2.VideoWriter(
		'./out/' + VideoName + '_Salganmore.avi',
			cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
			(w, h), isColor=False)
		while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame:
			_, frame = VideoCap.read()
        		frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame2 = frame.astype(np.uint8)
			saliency_map = predict(model, frame2)
			Out_frame = np.uint8(saliency_map)
			videoWriter.write(saliency_map)
		videoWriter.release()
		VideoCap.release()

if __name__ == "__main__":
	main()
