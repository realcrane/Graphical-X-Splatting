"""Extract a specific frame from a video and save it as an image.

Optionally produces a square center crop, extracts arbitrary [x, y, size] crop
sections (resized to 512x512), and draws coloured borders around them. Border
sizes are configurable: ``--border-thickness`` (rectangle outline, default 3) and
``--section-border-width`` (frame added around section images, default 5).
Use e.g. ``--border-thickness 15 --section-border-width 25`` for thick borders.

(This merges the former frame2image.py and video2fig_img.py into one tool.)
"""

import argparse
import cv2
import json
import os


def center_crop(image, crop_width, crop_height):
    """Center crop an image to the specified dimensions."""
    h, w = image.shape[:2]

    if crop_width > w or crop_height > h:
        raise ValueError(f"Crop dimensions ({crop_width}x{crop_height}) exceed image dimensions ({w}x{h})")

    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2

    return image[start_y:start_y+crop_height, start_x:start_x+crop_width]


def get_distinct_colors(n):
    """Generate n distinct BGR colors for visualizing crop sections."""
    import numpy as np
    colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Orange
        (255, 128, 0),    # Sky Blue
        (128, 255, 0),    # Spring Green
    ]

    # If we need more colors, generate them using HSV
    if n > len(colors):
        for i in range(n - len(colors)):
            hue = int((i * 180) / (n - len(colors)))
            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))

    return colors[:n]


def extract_crop_sections(image, crop_sections):
    """Extract multiple [x, y, size] square regions from an image."""
    h, w = image.shape[:2]
    cropped_regions = []

    for section in crop_sections:
        if len(section) != 3:
            raise ValueError(f"Each crop section must have exactly 3 values [x, y, size], got {section}")

        x, y, size = section

        # Validate crop region
        if x < 0 or y < 0 or x + size > w or y + size > h:
            raise ValueError(f"Crop section [{x}, {y}, {size}] is out of bounds for image of size {w}x{h}")

        cropped_region = image[y:y+size, x:x+size]
        cropped_regions.append(cropped_region)

    return cropped_regions


def extract_frame(video_path, frame_number, output_dir, experiment_name, output_filename=None,
                  do_crop=False, crop_sections=None, draw_borders=False,
                  border_thickness=3, section_border_width=5):
    """Extract a specific frame from a video file.

    Args:
        video_path: Path to the input video file
        frame_number: Frame number to extract (0-indexed)
        output_dir: Directory where the image will be saved
        experiment_name: Name of the experiment (used in output filename)
        output_filename: Optional custom filename. If None, will be auto-generated.
        do_crop: If True, creates a center crop of height×height (square)
        crop_sections: Optional list of [x, y, size] for extracting multiple specific regions
        draw_borders: If True, draws colored borders around crop sections and section images
        border_thickness: Line thickness for the rectangles drawn around crop sections
        section_border_width: Width of the coloured frame added around each section image

    Returns:
        Path(s) to the saved image file(s). Returns tuple or list depending on crop options used.
    """
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        raise ValueError(f"Error: Frame number {frame_number} is out of range. "
                        f"Video has {total_frames} frames (0-{total_frames-1})")

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Error: Could not read frame {frame_number} from video")

    # Create output directory with experiment name subfolder
    experiment_output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Generate output filename if not provided
    if output_filename is None:
        output_filename = f"{experiment_name}_frame_{frame_number:06d}.png"

    # Ensure filename has an extension
    if not any(output_filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
        output_filename += '.png'

    # Save the frame
    output_path = os.path.join(experiment_output_dir, output_filename)
    cv2.imwrite(output_path, frame)

    # Save center-cropped version if requested
    saved_paths = [output_path]

    # Determine which image to use as source for crop sections
    source_image = frame
    if do_crop:
        h, w = frame.shape[:2]
        crop_size = h  # Use height for both width and height (square crop)
        cropped_frame = center_crop(frame, crop_size, crop_size)
        source_image = cropped_frame  # Use center crop as source for sections

        # If draw_borders is enabled and we have crop sections, draw them on the center crop
        if draw_borders and crop_sections is not None:
            # Get distinct colors for each section
            colors = get_distinct_colors(len(crop_sections))

            # Draw rectangles on the center-cropped image
            cropped_frame_with_borders = cropped_frame.copy()
            for idx, section in enumerate(crop_sections):
                x, y, size = section

                # Coordinates are already relative to the source image (center crop in this case)
                # Only draw if the section is visible in the center crop
                if (x + size > 0 and y + size > 0 and
                    x < crop_size and y < crop_size):
                    # Clamp coordinates to image boundaries
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(crop_size, x + size)
                    y2 = min(crop_size, y + size)

                    cv2.rectangle(cropped_frame_with_borders, (x1, y1), (x2, y2),
                                colors[idx], thickness=border_thickness)

            cropped_frame = cropped_frame_with_borders

        # Generate cropped filename
        base_name, ext = os.path.splitext(output_filename)
        cropped_filename = f"{base_name}_crop_{crop_size}x{crop_size}{ext}"
        cropped_output_path = os.path.join(experiment_output_dir, cropped_filename)

        cv2.imwrite(cropped_output_path, cropped_frame)
        saved_paths.append(cropped_output_path)
    else:
        # If not doing center crop but drawing borders, draw on original image
        if draw_borders and crop_sections is not None:
            # Get distinct colors for each section
            colors = get_distinct_colors(len(crop_sections))

            # Draw rectangles on the original image
            frame_with_borders = frame.copy()
            for idx, section in enumerate(crop_sections):
                x, y, size = section
                cv2.rectangle(frame_with_borders, (x, y), (x + size, y + size),
                            colors[idx], thickness=border_thickness)

            # Save the original image with borders
            base_name, ext = os.path.splitext(output_filename)
            bordered_filename = f"{base_name}_with_borders{ext}"
            bordered_output_path = os.path.join(experiment_output_dir, bordered_filename)
            cv2.imwrite(bordered_output_path, frame_with_borders)
            saved_paths.append(bordered_output_path)

    # Save crop sections if requested
    if crop_sections is not None:
        cropped_regions = extract_crop_sections(source_image, crop_sections)
        base_name, ext = os.path.splitext(output_filename)

        # Get distinct colors if drawing borders
        colors = get_distinct_colors(len(crop_sections)) if draw_borders else None

        for idx, (region, section) in enumerate(zip(cropped_regions, crop_sections)):
            x, y, size = section

            # Resize the cropped section to 512x512
            region = cv2.resize(region, (512, 512), interpolation=cv2.INTER_LINEAR)

            # Add colored border frame around the cropped region if requested
            if draw_borders:
                region = cv2.copyMakeBorder(
                    region,
                    section_border_width, section_border_width,
                    section_border_width, section_border_width,
                    cv2.BORDER_CONSTANT,
                    value=colors[idx]
                )

            section_filename = f"{base_name}_section_{idx}_x{x}_y{y}_s{size}{ext}"
            section_output_path = os.path.join(experiment_output_dir, section_filename)
            cv2.imwrite(section_output_path, region)
            saved_paths.append(section_output_path)

    # Return appropriate result based on what was saved
    if len(saved_paths) == 1:
        return saved_paths[0]
    elif len(saved_paths) == 2 and do_crop and crop_sections is None:
        return tuple(saved_paths)  # Maintain backward compatibility
    else:
        return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description='Extract a specific frame from a video file and save it as an image.'
    )
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('frame_number', type=int, help='Frame number to extract (0-indexed)')
    parser.add_argument('output_dir', type=str, help='Directory where the image will be saved')
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment (used in output filename)')
    parser.add_argument('--output-filename', type=str, default=None,
                        help='Custom output filename (optional). If not provided, auto-generated '
                             'as {experiment_name}_frame_{frame_number}.png')
    parser.add_argument('--crop', action='store_true',
                        help='Create a center crop of height×height (square). Saves both original '
                             'and center-cropped versions.')
    parser.add_argument('--crop-sections', type=str, default=None,
                        help='JSON string defining crop sections: "[[x1,y1,size1], ...]". Each '
                             'section is [x, y, size] where (x,y) is top-left and size is width/height.')
    parser.add_argument('--draw-borders', action='store_true',
                        help='Draw colored borders around crop sections and add colored frames to '
                             'section images.')
    parser.add_argument('--border-thickness', type=int, default=3,
                        help='Line thickness for rectangles drawn around crop sections (default: 3).')
    parser.add_argument('--section-border-width', type=int, default=5,
                        help='Width of the coloured frame added around each section image (default: 5).')

    args = parser.parse_args()

    try:
        # Parse crop sections if provided
        crop_sections = None
        if args.crop_sections:
            crop_sections = json.loads(args.crop_sections)
            if not isinstance(crop_sections, list):
                raise ValueError("crop-sections must be a list of [x, y, size] values")

        result = extract_frame(
            args.video_path,
            args.frame_number,
            args.output_dir,
            args.experiment_name,
            args.output_filename,
            args.crop,
            crop_sections,
            args.draw_borders,
            args.border_thickness,
            args.section_border_width,
        )

        # Handle different return types
        if isinstance(result, list):
            print(f"Successfully saved frame {args.frame_number} to:")
            for path in result:
                print(f"  - {path}")
        elif isinstance(result, tuple):
            output_path, cropped_path = result
            print(f"Successfully saved frame {args.frame_number} to: {output_path}")
            print(f"Successfully saved cropped version to: {cropped_path}")
        else:
            output_path = result
            print(f"Successfully saved frame {args.frame_number} to: {output_path}")

        # Print video info
        cap = cv2.VideoCapture(str(args.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video info: {total_frames} total frames, {fps:.2f} FPS")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
