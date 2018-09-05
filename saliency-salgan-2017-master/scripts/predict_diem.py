import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils2 import *
from constants import *
from models.model_bce import ModelBCE

Video_dir = '/home/s/re/CITIUS'


def main():
	# Create network
	model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=1)
	# Here need to specify the epoch of model sanpshot
	load_weights(model.net['output'], path='gen_', epochtoload=90)
	# Here need to specify the path to images and output path
	namelist = os.listdir(Video_dir)
	for videonameind in namelist:
		VideoName = videonameind[:-4]
		VideoCap = cv2.VideoCapture(Video_dir + '/' + videonameind)
		#print(os.path.join(Video_dir,VideoName, VideoName))
		VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		print VideoSize
		VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
		print('New video: %s with %d frames and size of (%d, %d)' % (
		VideoName, VideoFrame, VideoSize[1], VideoSize[0]))
		out = cv2.VideoWriter('./out/' + VideoName + '.avi',
				cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,VideoSize, isColor=False)
		while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame:
			_, frame = VideoCap.read()
			frame2= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame2 = frame2.astype(np.uint8)
			saliency_map = predict(model, frame2)
			Out_frame = np.uint8(saliency_map)
			out.write(Out_frame)
		out.release()
        VideoCap.release()

if __name__ == "__main__":
	main()
