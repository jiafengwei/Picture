from Salicon import Salicon
import cv2
from PIL import Image
import numpy as np
import time
#import h5py
import random
import os

resize_shape = (800, 600)

tofile_img = 'out.jpg'

VideoNameFile = 'Validationlist.txt'  # choose the data
Video_dir = './dataset'
SaveFile = './model/'
Summary_dir = './summary'



def main():
    sal = Salicon()

    videofile = open(VideoNameFile, 'r')
    allline = videofile.readlines()
    VideoIndex_list = []
    VideoName_list = []
    for line in allline:
        lindex = line.index('\t')
        VideoIndex = int(line[:lindex])
        VideoName = line[lindex + 1:-1]
        VideoName = VideoName[0:len(VideoName)-1]
        VideoIndex_list.append(VideoIndex)
        VideoName_list.append(VideoName)
    VideoNum = len(VideoName_list)
    epochsort = np.arange(0, VideoNum)

    for v in epochsort:
        VideoIndex = VideoIndex_list[v]
        VideoName = VideoName_list[v]
        VideoCap = cv2.VideoCapture(os.path.join(Video_dir,VideoName, VideoName + '.mp4'))
        #print(os.path.join(Video_dir,VideoName, VideoName))
        VideoSize = (int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        VideoFrame = int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        assert VideoSize[0] == int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) and VideoSize[1] == int(
            VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) and VideoFrame == int(VideoCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print('New video: %s (%d) with %d frames and size of (%d, %d)' % (
        VideoName, VideoIndex, VideoFrame, VideoSize[1], VideoSize[0]))
        
        
        fps = float(VideoCap.get(cv2.cv.CV_CAP_PROP_FPS))
        out = cv2.VideoWriter(os.path.join(Video_dir,VideoName, VideoName + '_out.avi'),
                        cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), fps,
                        resize_shape, isColor=True)
        while VideoCap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) < VideoFrame:
            start_time = time.time()
            _, frame = VideoCap.read()
            print('frame:'+str(VideoCap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))+' of '+str(VideoFrame))
            frame = frame.astype(np.uint8)
            print(str(frame.shape))
            frame = cv2.resize(frame, resize_shape)
            print(str(frame.shape))
            #frame = frame / 255.0 * 2 - 1
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = cv2.resize(frame, fine_resize_shape)

            map = sal.compute_saliency(frame)
            dis = np.amax(map) - np.amin(map)
            map = (map - np.amin(map)) / dis * 255
            # w, h = map.shape
            # ret = np.empty((w, h, 3), dtype=np.uint8)
            # ret[:, :, 0] = map
            ret = cv2.resize(map, resize_shape)
            ret = ret[..., np.newaxis]
            zerosnp = np.zeros_like(ret)
            ret = np.concatenate((zerosnp, zerosnp, ret), axis=2)
            ret = np.uint8(ret)
            out.write(ret)
            duration = float(time.time() - start_time)
            print('Frame time %f' % (duration))
        out.release()
        VideoCap.release()


if __name__ == '__main__':
    main()
