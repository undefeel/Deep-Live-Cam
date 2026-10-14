from typing import Any, List

import os

import modules.processors.frame.core
from modules import hand_portal, imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.typing import Frame, Face

NAME = "DLC.HAND-PORTAL"


def pre_check() -> bool:
    if hand_portal.ensure_model() is None:
        update_status("Hand landmarker model is unavailable.", NAME)
        return False
    return True


def pre_start() -> bool:
    return True


def process_frame(source_face: Face | None, temp_frame: Frame, **kwargs: Any) -> Frame:
    """Applies the hand portal effect to a frame; source_face is unused."""
    return hand_portal.apply_hand_portal(temp_frame)


def process_frame_v2(temp_frame: Frame, **kwargs: Any) -> Frame:
    """Processes a frame without source face (used by live webcam preview)."""
    return hand_portal.apply_hand_portal(temp_frame)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(
                f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            print(
                f"{NAME}: Warning: Failed to read frame {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(
    source_path: str | None, target_path: str, output_path: str
) -> None:
    """Processes a single image file."""
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"{NAME}: Hand portal image saved to {output_path}")


def process_video(
    source_path: str | None, temp_frame_paths: List[str]
) -> None:
    """Processes video frames using the frame processor core."""
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
