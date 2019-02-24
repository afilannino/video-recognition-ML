import glob
import os

import matplotlib.pyplot as plt
import tqdm

from utility.utility import retrieve_videoobject_subsets, project_root
from pytorch_liteflownet.run import run
from preprocessing.f2i import Flow


def main():
    subsets = retrieve_videoobject_subsets(['train', 'validation'])
    generate_flow_frames(subsets[0])
    generate_flow_frames(subsets[1])


def generate_flow_frames(subset, skip_existent=False):
    # Initializing progress bar
    length = len(subset)
    print('Starting flow frames generation')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for video in subset:
        frames_folder_root = os.path.join(project_root(), 'data', 'UCF-101-frames', video.label, 'v_' + video.label +
                                          '_' + video.group + '_' + video.clip)
        flowframes_folder_root = os.path.join(project_root(), 'data', 'UCF-101-flowframes', video.label, 'v_' +
                                              video.label + '_' + video.group + '_' + video.clip)
        frame_folder_name = frames_folder_root + '_frames'
        flow_frame_folder_name = flowframes_folder_root + '_flowframes'
        if not os.path.exists(flow_frame_folder_name):
            os.makedirs(flow_frame_folder_name)
        if not os.path.exists(frame_folder_name):
            raise Exception('You have to generate frames first and then create the optical flow frames!')

        # Skip folder if frames have been already generated
        if skip_existent:
            first_flow_frame_filename = os.path.join(flow_frame_folder_name, 'flowframe-0001.jpg')
            if os.path.exists(first_flow_frame_filename):
                progress_bar.update(1)
                continue

        frames = glob.glob(os.path.join(frame_folder_name, 'frame-*.jpg'))
        generate_and_store_flowframes(frames, flow_frame_folder_name)
        progress_bar.update(1)

    progress_bar.close()


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


if __name__ == '__main__':
    main()
