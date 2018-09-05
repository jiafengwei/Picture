import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
VideoNameFile = 'Testlist.txt'  # choose the data
Video_dir = '/home/s/dataset/SFU'

def test(path_to_images, path_output_maps, model_to_test=None):
	list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
	# Load Data
	list_img_files.sort()
	for curr_file in tqdm(list_img_files, ncols=20):
		print os.path.join(path_to_images, curr_file + '.jpg')
		img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		saliency_map = predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)


def main():
	# Create network
	model = ModelBCE(352,288, batch_size=1)
	# Here need to specify the epoch of model sanpshot
	load_weights(model.net['output'], path='gen_', epochtoload=90)
	# Here need to specify the path to images and output path
	namelist = os.listdir(Video_dir)
	for videonameind in namelist:
		VideoName = videonameind[:-4]
		VideoCap = cv2.VideoCapture(Video_dir + '/' + videonameind)
		#print(os.path.join(Video_dir,VideoName, VideoName))
		VideoSize = (int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
		VideoFrame = int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		print('New video: %s with %d frames and size of (%d, %d)' % (
		VideoName, VideoFrame, VideoSize[1], VideoSize[0]))
		fps = float(VideoCap.get(cv2.cv.CV_CAP_PROP_FPS))
		out = cv2.VideoWriter('./outsfu/' + VideoName + '_Salgan.avi',
				cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), fps,(352, 288), isColor=False)
		while VideoCap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) < VideoFrame:
			_, frame = VideoCap.read()
			frame = frame.astype(np.uint8)
			saliency_map = predict(model, frame)
			Out_frame = np.uint8(saliency_map)
			out.write(Out_frame)
        out.release()
        VideoCap.release()

if __name__ == "__main__":
	main()
