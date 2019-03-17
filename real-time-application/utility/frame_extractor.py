import cv2
import numpy as np
import os
import tqdm
import matplotlib as plt

from external.flow_to_image.f2i import Flow
# from external.pytorch_liteflownet.run import run as generate_optical_flow
from utility.utility import limit_frames_number, project_root


class FrameExtractor(object):

    def __init__(self):
        self.fps = 25
        self.seconds_per_segment = 2
        self.frames_per_segment = 10
        self.displacement = 1
        self.optical_flow_converter = Flow()
        # self.optical_flow_extractor = generate_optical_flow
        self.current_segment = 0
        self.starting_displacement = 0
        self.use_displacement = False

    def increase_current_segment(self):
        self.current_segment += 1

    def set_starting_displacement(self, value):
        self.starting_displacement = value

    def set_second_per_segment(self, sps):
        self.seconds_per_segment = sps

    def set_frames_per_segment(self, fpseg):
        self.frames_per_segment = fpseg

    def frames_generator(self, video_file):
        video_cap = cv2.VideoCapture(video_file)
        segment_to_yield = round(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / (video_cap.get(cv2.CAP_PROP_FPS) * 2))
        video_cap.release()
        counter = 0

        while True:
            if counter >= segment_to_yield:
                yield 'STOP', 'STOP'

            rgb_frames, flow_frames = self.extract_frames(video_file, self.use_displacement)
            if self.use_displacement:
                self.increase_current_segment()

            additional_info = [self.use_displacement, self.current_segment]
            self.use_displacement = not self.use_displacement
            counter += 1
            yield rgb_frames, flow_frames, additional_info

    def extract_frames(self, video_file, displacement):
        if not os.path.exists(video_file):
            print('Cannot fine video file')
            raise Exception('File video not found')

        rgb_frames = self.generate_rgb_frames(video_file, displacement)
        # flow_frames = self.generate_flow_frames(rgb_frames)
        return rgb_frames, 0  # flow_frames  #TODO

    def generate_rgb_frames(self, video_file, displacement):
        frames = []

        starting_time = self.frames_per_segment * self.current_segment
        if displacement:
            starting_time += self.displacement

        video_cap = cv2.VideoCapture(video_file)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, starting_time)

        i = 0
        while i < video_cap.get(cv2.CAP_PROP_FPS)*self.seconds_per_segment:
            ret, frame = video_cap.read()
            if ret is False:
                break
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            frames.append(frame)
            i += 1
        video_cap.release()
        # cv2.destroyAllWindows()
        frames = limit_frames_number(frames, self.frames_per_segment)
        return frames

    def generate_flow_frames(self, rgb_frames):
        flow_frames = []
        for i in range(len(rgb_frames)):
            if i == len(rgb_frames) - 1:
                break

            # generate optical flow
            temp_path = os.path.join(project_root(), 'data', 'result', 'image.flo')
            generate_optical_flow('default', rgb_frames[i], rgb_frames[i + 1], temp_path)

            # save it as image
            img = self.optical_flow_converter.convertFromFile(temp_path)
            flow_frames.append(flow_frames)

        flow_frames = limit_frames_number(flow_frames, self.frames_per_segment)
        return flow_frames


def generate_and_store_flowframes(frames, flow_frame_folder_name):
    for i in range(len(frames)):
        if i == len(frames) - 1:
            break

        # generate optical flow
        temp_path = os.path.join(project_root(), 'data', 'result', 'image.flo')
        run('default', frames[i], frames[i+1], temp_path)

        # save it as image
        flow = Flow()
        img = flow.convertFromFile(temp_path)
        to_save = os.path.join(flow_frame_folder_name, 'flowframe-{:0>4d}.png'.format(i))
        plt.imsave(to_save, img)
