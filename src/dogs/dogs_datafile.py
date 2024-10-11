import subprocess
from pathlib import Path

from tqdm.notebook import tqdm
import pandas as pd

from src.dogs.dog_video import DogVideo

class DogsDatafile:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        df = pd.read_parquet("hf://datasets/Corran/pexelvideos/PexelVideos.parquet.gzip")
        """
        Output looks like this:
        0          https://www.pexels.com/video/mother-and-two-kids-sitting-on-red-couch-7603862/
        1                            https://www.pexels.com/video/a-girl-playing-trumpet-7603863/
        2    https://www.pexels.com/video/girl-is-posing-while-bringing-her-trumpet-case-7603874/
        3                   https://www.pexels.com/video/a-girl-opening-the-trumpet-case-7603875/
        4    https://www.pexels.com/video/girls-leaning-on-a-white-wall-with-eyes-closed-7603876/
        Name: loc, dtype: object

        => Parse the trailing ints.
        """
        df["pexels_video_id"] = df["loc"].str.extract(r'-(\d+)/$')
        dog_df = df[df["title"].str.lower().str.contains("dog")]
        assert len(dog_df) > 0, "No dogs found in the dataset."
        self.df = dog_df

    def download_videos(self):
        for video_id in tqdm(self.df["pexels_video_id"].tolist()):
            self.download_video_id(video_id)

    def download_video_id(self, video_id, remove_if_exists=False) -> bool:
        output_path = self.data_dir / "pexels-videos" / f"{video_id}.mp4"
        if Path(output_path).exists():
            if remove_if_exists:
                Path(output_path).unlink()
                print(f"Removed existing file {output_path}.")
            else:
                print(f"File {output_path} already exists.")
                return True
        system_command = f'wget -O "{output_path}" $(curl -Ls -o /dev/null -w %{{url_effective}} "https://www.pexels.com/download/video/{video_id}/")'
        print(system_command)
        # run command, don't let output print to console, and return True if successful.
        result = subprocess.run(system_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"Downloaded video {video_id} to {output_path}.")
            return True
        else:
            print(f"Failed to download video {video_id}.")
            # print console.
            print(result.stdout.decode())
            print(result.stderr.decode())
        return False

    def crop_dog_videos(self):
        for video_id in tqdm(self.df["pexels_video_id"].tolist()):
            self.crop_video_id(video_id)

    def crop_video_id(self, video_id):
        video_path = self.data_dir / "pexels-videos" / f"{video_id}.mp4"
        if video_path.exists():
            video = DogVideo(video_path)
            if not video.cropped_path.exists():
                if video.are_most_frames_the_same_dog():
                    video.crop_dog_video()

