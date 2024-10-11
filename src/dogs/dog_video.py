from pathlib import Path
import math

from PIL import Image, ImageDraw
import cv2

from src.dogs.dog_detector import DogDetector, BoundingBox

class DogVideo:
    def __init__(self, path: Path):
        assert path.exists(), f"Video file not found: {path}"
        self.path = path

    def play(self):
        print(f"Playing video: {self.path}")

    def to_pil_frames(self, max_frames: int | None = 16, override: bool = False) -> list[Image.Image]:
        if hasattr(self, "frames") and not override:
            return self.frames

        video = cv2.VideoCapture(str(self.path))
        frames = []

        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for_each_keep_drop_n = max(1, math.floor(num_frames / max_frames - 1))

        def read_and_append() -> bool:
            success, image = video.read()
            if not success:
                return False
            image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            image.thumbnail((512, 512)) # operates in place.
            frames.append(image)
            return True

        def just_read() -> bool:
            return video.read()[0]

        while read_and_append():
            for _ in range(for_each_keep_drop_n):
                if not just_read():
                    break

        video.release()

        assert len(frames) > 0, f"No frames found in video: {self.path}"
        self.frames = frames
        return frames

    def look_for_dogs(self, override: bool = False) -> list[list[BoundingBox]]:
        if hasattr(self, "detection_results") and not override:
            return self.detection_results
        
        frames = self.to_pil_frames(max_frames=16)

        detector = DogDetector()
        detection_results = detector.detect_objects(
            images=frames,
            object_name="dog",
            batch_size=16,
        )

        self.detection_results = detection_results
        return self.detection_results.copy()

    def draw_dogs(self, frame_idx: int) -> Image.Image:
        """
        Get the frame,
        get the detection (if any),
        and draw the bounding boxes.
        """
        frames = self.to_pil_frames(max_frames=None)
        detection_results = self.look_for_dogs()

        frame_copy = frames[frame_idx].copy()
        dog_boxes = detection_results[frame_idx]

        if not dog_boxes:
            return frame_copy

        for box in dog_boxes:
            draw = ImageDraw.Draw(frame_copy)
            box_width = int(frame_copy.height / 100)
            font_size = int(frame_copy.height / 10)
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline="red", width=box_width)
            # draw object name inside the box too (LARGE!)
            draw.text((box.x1 + 2 * box_width, box.y1), box.name, fill="red", font_size=font_size)

        return frame_copy

    def are_most_frames_the_same_dog(self, most: float = 0.6) -> bool:
        frames = self.to_pil_frames(max_frames=16)
        detection_results = self.look_for_dogs()

        last_lone_dog_box: BoundingBox | None = None
        count = 0

        for dog_boxes in detection_results:
            if len(dog_boxes) != 1:
                continue

            # We've got a lone dog.
            if last_lone_dog_box is None:
                last_lone_dog_box = dog_boxes[0]
                count += 1
                continue

            # We've got an updated dog position.
            # Count if overlap.
            # Update position regardless.
            if last_lone_dog_box.overlaps(dog_boxes[0]):
                count += 1

            last_lone_dog_box = dog_boxes[0]

        # print(f"[dog_video.py:debug] {count=} / {len(frames)=} frames have the same dog.")
        return count / len(frames) >= most

    @staticmethod
    def interpolate_bounding_boxes(
        detection_results: list[list[BoundingBox]],
    ) -> list[BoundingBox]:
        """
        For every non lone-dog frame, interpolate between the preceding and
        following lone-dog frames.

            If not applicable (e.g. first frame), use the coords from the nearest long-dog frame.
        """
        last_lone_dog_frame = None
        frames_since_last = 0
        interpolated_boxes = []
        def catch_up(next_box: BoundingBox | None) -> None:
            for i in range(frames_since_last):
                if last_lone_dog_frame is None:
                    interpolated_boxes.append(next_box)
                elif next_box is None:
                    interpolated_boxes.append(last_lone_dog_frame)
                else:
                    interpolated_boxes.append(
                        last_lone_dog_frame.interpolate(
                            other=next_box,
                            slider=(i+1) / (frames_since_last+1), # 50% if 1 frame missed.
                        )
                    )
        for detection_boxes in detection_results:
            if len(detection_boxes) != 1:
                # Note a lone dog box.
                frames_since_last += 1
                continue
            # IS a lone-dog box!
            catch_up(detection_boxes[0])
            frames_since_last = 0
            interpolated_boxes.append(detection_boxes[0])
            last_lone_dog_frame = detection_boxes[0]
        assert len(interpolated_boxes) > 0, "No dogs?"
        catch_up(next_box=None)
        assert len(detection_results) == len(interpolated_boxes), f"{len(detection_results)=} != {len(interpolated_boxes)=}"
        return interpolated_boxes

    def crop_video_frames(self, letterbox_to: tuple[int, int] | None = None) -> list[Image.Image]:
        frames = self.to_pil_frames(max_frames=16)
        detection_results = self.look_for_dogs()
        crop_boxes: list[BoundingBox] = DogVideo.interpolate_bounding_boxes(detection_results)
        cropped_frames = []
        for frame, box in zip(frames, crop_boxes):
            # Crop the frame
            cropped_frame = frame.crop((box.x1, box.y1, box.x2, box.y2))
            if letterbox_to is not None:
                cropped_frame = DogVideo.letterbox_image(cropped_frame, letterbox_to)
            cropped_frames.append(cropped_frame)
        
        return cropped_frames

    @staticmethod
    def letterbox_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
        """
        Letterbox an image and resize to the given dimensions.
        """
        iw, ih = image.size
        w, h = box
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.LANCZOS)
        new_image = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        return new_image

    @property
    def cropped_path(self) -> Path:
        return self.path.parent / "dog-videos" / self.path.stem

    def crop_dog_video(self) -> Path:
        output_dir = self.cropped_path
        output_dir.mkdir(exist_ok=False, parents=False)

        frames_to_write = self.crop_video_frames((512, 512))

        for ix, frame in enumerate(frames_to_write):
            out_path = output_dir / f"frame_{ix:02d}.png"
            frame.save(out_path, format="PNG")

        return output_dir
