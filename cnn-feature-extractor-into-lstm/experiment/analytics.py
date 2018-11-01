import os
import glob
import csv

from utility.utility import project_root, retrieve_videoobject_subsets

project_root = project_root()


def main():
    #create_frames_number_csv()
    evaluate_frames()


def create_frames_number_csv():
    videoobject_subsets = retrieve_videoobject_subsets(['train', 'validation', 'test'])

    header = ['VIDEO', 'N_FRAMES']
    with open(os.path.join(project_root, 'data', 'result', 'n_frames_per_video.csv'), 'w', newline='\n') as frame_csv:
        frame_csv_writer = csv.writer(frame_csv, delimiter=',')
        frame_csv_writer.writerow(header)

        for videoobject_subset in videoobject_subsets:

            for video in videoobject_subset:

                video_name = 'v_' + video.label + '_' + video.group + '_' + video.clip
                frame_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label, video_name + '_frames')

                # Retrieve frames for this video
                frames = glob.glob(os.path.join(frame_folder_name, 'frame-*.jpg'))

                frame_csv_writer.writerow([video_name, len(frames)])


def evaluate_frames(size_limit=100):
    n_video_under_size_limit = 0
    frame_sum = 0
    video_sum = 0
    missing_frames = 0
    min_frames = 14000
    max_frames = 0

    with open(os.path.join(project_root, 'data', 'result', 'n_frames_per_video.csv'), 'r', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            video_sum += 1
            frame_sum += int(row[1])
            if int(row[1]) < size_limit:
                n_video_under_size_limit += 1
                missing_frames += size_limit - int(row[1])
            if int(row[1]) < min_frames:
                min_frames = int(row[1])
            if int(row[1]) > max_frames:
                max_frames = int(row[1])

    header = ['STATS', 'VALUE']
    with open(os.path.join(project_root, 'data', 'result', 'statistics.csv'), 'w', newline='\n') as stat_csv:
        stat_csv_writer = csv.writer(stat_csv, delimiter=',')
        stat_csv_writer.writerow(header)
        # size_limit independent
        stat_csv_writer.writerow(['video_sum', video_sum])
        stat_csv_writer.writerow(['frames_sum', frame_sum])
        stat_csv_writer.writerow(['average_frames_per_video', frame_sum / video_sum])
        stat_csv_writer.writerow(['video_with_min_frames', min_frames])
        stat_csv_writer.writerow(['video_with_max_frames', max_frames])
        # size_limit dependent
        stat_csv_writer.writerow(['video_with_less_than_size_limit_frames', n_video_under_size_limit])
        stat_csv_writer.writerow(['missing_frames', missing_frames])
        stat_csv_writer.writerow(['average_missing_frames', missing_frames / n_video_under_size_limit])


if __name__ == '__main__':
    main()
