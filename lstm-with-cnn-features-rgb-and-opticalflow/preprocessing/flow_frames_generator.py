import glob
import os

import cv2
import numpy as np
import tqdm

from utility.utility import retrieve_videoobject_subsets, limit_frames_size, project_root


def main():
    generate_flow_frames(['train', 'validation'], sequence_length=31)


def generate_flow_frames(subsets, sequence_length=31, skip_existent=False):
    # Retrieve video's subsets
    videoobject_subsets = retrieve_videoobject_subsets(subsets)

    # Initializing progress bar
    length = 0
    for videoobject_subset in videoobject_subsets:
        length += len(videoobject_subset)
    print('Starting flow frames generation')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for videoobject_subset in videoobject_subsets:

        for video in videoobject_subset:
            video_base = os.path.join(project_root(), 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip)
            frame_folder_name = video_base + '_frames'
            flow_frame_folder_name = video_base + '_flowframes'
            if not os.path.exists(flow_frame_folder_name):
                os.mkdir(flow_frame_folder_name)
            if not os.path.exists(frame_folder_name):
                raise Exception('You have to generate frames first and then create the optical flow frames!')

            # Skip folder if frames have been already generated
            if skip_existent:
                first_flow_frame_filename = os.path.join(flow_frame_folder_name, 'flowframe-0001.jpg')
                if os.path.exists(first_flow_frame_filename):
                    progress_bar.update(1)
                    continue

            frames = glob.glob(os.path.join(frame_folder_name, 'frame-*.jpg'))
            frames = limit_frames_size(frames, sequence_length)

            # Generate flow frames
            frame1 = cv2.imread(frames[0])
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            for i in range(len(frames) - 1):
                if i == 0:
                    continue
                flow_frame_file_output = os.path.join(flow_frame_folder_name, 'flowframe-{:04d}.jpg'.format(i))
                next_frame_file = os.path.join(frame_folder_name, 'frame-{:04d}.jpg'.format(i+1))

                frame2 = cv2.imread(next_frame_file)
                nextt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, nextt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # optical_flow = cv2.DualTVL1OpticalFlow_create()
                # flow = optical_flow.calc(prvs, nextt, None)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                cv2.imshow('Optical flow during flow frames creation', bgr)
                cv2.waitKey(1)
                cv2.imwrite(flow_frame_file_output, bgr)
                prvs = nextt

            progress_bar.update(1)

    cv2.destroyAllWindows()
    progress_bar.close()


if __name__ == '__main__':
    main()
