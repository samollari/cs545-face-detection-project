import sys
import zipfile
import numpy as np
from PIL import Image
import io
import re
import cv2
from functools import partial, wraps
import math
import os
from facenet_pytorch import MTCNN
import argparse

from tqdm import tqdm  # Added import

# Initialize MTCNN detector
detector = MTCNN(device="cuda")


def fit_512(image):
    h, w, _ = image.shape
    max_dim = max(h, w)

    if max_dim > 512:
        scale_factor = 512 / max_dim
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return image


def y_channel(image):
    return image[:, :, 0]


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)


def yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def get_hist(image):
    """
    Compute the histogram of the Y channel of the image.
    Args:
        image (ndarray): Input image in YUV color space.
    Returns:
        tuple: Histogram and bin edges.
    """
    hst, bins = np.histogram(y_channel(image), bins=np.arange(256))
    return hst, bins


def get_cdf(hst):
    """
    Compute the cumulative distribution function (CDF) of the histogram.
    Args:
        hst (ndarray): Histogram values.
        bins (ndarray): Bin edges.
    Returns:
        ndarray: CDF values.
    """
    cdf = np.cumsum(hst)
    return cdf


def get_hist_cdf(image):
    """
    Compute both the histogram and CDF of the Y channel of the image.
    Args:
        image (ndarray): Input image in YUV color space.
    Returns:
        tuple: Histogram, bin edges, and CDF values.
    """
    hst, bins = get_hist(image)
    cdf = get_cdf(hst)
    return hst, bins, cdf


def copy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            copied_args = (np.copy(args[0]),) + args[1:]
            return func(*copied_args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def grayscale(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            grayscale_img = y_channel(args[0])
            return func(grayscale_img, *args[1:], **kwargs)  # Pass grayscale image
        return func(*args, **kwargs)

    return wrapper


@copy
def dither(image, scale=2.0):
    y = y_channel(image)
    image[:, :, 0] = np.clip(y + np.random.normal(scale=scale, size=y.shape), 0, 255)
    return image


@copy
def basic_contrast(image):
    y = y_channel(image)
    minval = np.min(y)
    maxval = np.max(y)
    # print(minval, maxval)

    scale_factor = 255 / (maxval - minval)
    # print(scale_factor)
    image[:, :, 0] = ((y - minval) * scale_factor).clip(0, 255)
    return image


@copy
def hist_eq(image, cdf):
    h, w, _ = image.shape

    def histogram_equalization(cdf, sizeprod, val):
        return math.floor((cdf[val - 1] if val != 0 else cdf[val]) * 255 / sizeprod)

    vec_hst_eq = np.vectorize(partial(histogram_equalization, cdf, h * w))
    image[:, :, 0] = vec_hst_eq(y_channel(image))
    return image


def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 5, 10)


def agressive_denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)


@grayscale
def edge_detect(image):
    return cv2.Canny(image, 100, 200)


def detect_face(image, detector: MTCNN):
    faces = detector.detect(rgb(image))[0]
    # print(faces)
    # return None
    if faces is None or len(faces) == 0:
        return None
    # Get the bounding box of the first detected face
    x, y, width, height = np.floor(faces[0])
    # Ensure coordinates are within bounds
    x, y = max(0, x), max(0, y)
    return x, y, width, height


def crop_to_face(image, box):
    # Get bounding box of the first detected face
    x, y, width, height = box

    # Ensure coordinates are non-negative
    x, y = max(0, x), max(0, y)

    # Calculate the center of the bounding box
    center_x = x + width / 2
    center_y = y + height / 2

    # Determine the side length of the square (use the larger dimension)
    side_length = max(width, height)

    # Calculate the top-left corner of the square
    square_x = int(center_x - side_length / 2)
    square_y = int(center_y - side_length / 2)
    square_width = int(side_length)
    square_height = int(side_length)

    # Get image dimensions
    img_h, img_w, _ = image.shape

    # Adjust square coordinates to stay within image bounds
    square_x = max(0, square_x)
    square_y = max(0, square_y)
    # Ensure the bottom-right corner doesn't exceed image dimensions
    if square_x + square_width > img_w:
        square_width = img_w - square_x
    if square_y + square_height > img_h:
        square_height = img_h - square_y
    # Recalculate side length if adjusted, ensuring it remains square
    side_length = min(square_width, square_height)
    square_width = side_length
    square_height = side_length
    # Adjust x,y again if width/height changed due to bottom/right boundary
    if square_x + square_width > img_w:
        square_x = img_w - square_width
    if square_y + square_height > img_h:
        square_y = img_h - square_height

    # Crop the original resized YUV image to the square
    cropped = image[
        square_y : square_y + square_height, square_x : square_x + square_width
    ]

    return dither(basic_contrast(cropped))


def preferred_face(a, b):
    if a and b:
        _, _, a_w, a_h = a
        _, _, b_w, b_h = b

        # Calculate the area of the detected faces
        a_area = a_w * a_h
        b_area = b_w * b_h

        # Choose the face with the larger area
        return a if a_area > b_area else b
    elif a:
        return a
    elif b:
        return b
    else:
        return None


def run_pipeline(
    file_name, file_contents, output_dir, dry_run=False
):  # Added dry_run parameter
    global detector
    """
    Process a file from the zip archive and save the result.
    Args:
        file_name (str): The name of the file.
        file_contents (bytes): The contents of the file.
        output_dir (str): The directory to save the processed image.
        dry_run (bool): If True, skip processing and just print the file name.
    """
    if dry_run:  # Added dry run check
        # print(f"[Dry Run] Would process file: {file_name}")
        return

    try:
        # Load the image using PIL and convert it to a numpy array
        image = Image.open(io.BytesIO(file_contents))
        image = np.array(image)

        # Convert the image to YUV color space
        image = np.asarray(yuv(image))

        image = fit_512(image)

        mid_image = remove_noise(dither(basic_contrast(image)))
        # mid_image = dither(basic_contrast(image))
        # result = mid_image

        init_face = detect_face(mid_image, detector)

        # cropped = crop_to_face(mid_image, detector, file_name)

        # Pass the potentially cropped image to get_hist_cdf
        _, _, mid_cdf = get_hist_cdf(mid_image)
        # print("hst", hst)
        # print("bins", bins)
        # print("cdf", cdf)

        full_result = dither(basic_contrast(hist_eq(mid_image, mid_cdf)))
        # result = hist_eq(mid_image, cdf)

        full_face = detect_face(full_result, detector)

        preferred = preferred_face(init_face, full_face)

        result = full_result
        if preferred:
            result = crop_to_face(full_result, preferred)

        edges = edge_detect(result)

        # Construct the output file path
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_processed.png")
        edge_output_path = os.path.join(output_dir, f"{base_name}_edges.png")

        # Convert back to BGR for saving
        image_to_save = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)

        # Save the processed image
        cv2.imwrite(output_path, image_to_save)
        cv2.imwrite(edge_output_path, edges)
        # print(f"Saved processed image to: {output_path}")

    except Exception as e:
        # Log the full traceback for better debugging
        import traceback

        print(f"Error processing file '{file_name}': {e}\n{traceback.format_exc()}")


def get_task(filename, zip_ref, output_dir, dry_run):
    """
    Create a task for processing a file from the zip archive.
    Args:
        file_info (ZipInfo): Information about the file in the zip archive.
        zip_ref (ZipFile): The zip file reference.
        output_dir (str): The directory to save the processed image.
        dry_run (bool): If True, skip processing and just print the file name.
    Returns:
        tuple: A tuple containing the file name, file contents, output directory, and dry run flag.
    """
    with zip_ref.open(filename) as file:
        return (
            filename,
            file.read(),
            output_dir,
            dry_run,
        )  # Return the task as a tuple


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process images from a zip file.")
    parser.add_argument("zip_path", help="Path to the zip file containing images.")
    parser.add_argument(
        "regex",
        nargs="?",
        default=None,
        help="Optional regex to filter files within the zip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without processing images.",
    )  # Added dry-run argument
    args = parser.parse_args()

    zip_path = args.zip_path
    file_regex = args.regex
    if file_regex:
        print(f"Using regex: {file_regex}")
    dry_run = args.dry_run  # Get dry_run flag
    num_workers = 4  # Define number of workers

    # Create output directory based on zip file name
    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    output_dir = os.path.join(os.getcwd(), zip_basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to directory: {output_dir}")
    if dry_run:
        print("--- Performing DRY RUN ---")

    # A generator asynchronously creating tasks for processing files
    # try:

    zip_ref = zipfile.ZipFile(zip_path, "r")

    def get_tasks(files_to_process):
        for file in files_to_process:
            yield get_task(file, zip_ref, output_dir, dry_run)

    files_to_process = [
        file_info.filename
        for file_info in zip_ref.infolist()
        if not file_info.is_dir()
        and (not file_regex or re.match(file_regex, file_info.filename))
    ]
    print(f"Found {len(files_to_process)} files matching the criteria in the zip file.")

    # Create a directory in the output for each class
    classes = set()
    for file in tqdm(files_to_process, desc="Detecting classes", unit="file"):
        class_name = os.path.dirname(file)
        classes.add(class_name)
    print(f"{len(classes)} classes found")
    for class_name in tqdm(classes, desc="Creating directories", unit="class"):
        class_dir = os.path.join(output_dir, class_name)
        if not dry_run:
            os.makedirs(class_dir, exist_ok=True)
        # print(f"Created directory for class: {class_name}")
    if len(classes) == 0:
        print("No classes found in the zip file.")
        sys.exit(0)

    # print(f"\nStarting processing with {num_workers} workers...")

    # with tqdm(
    #     total=len(files_to_process), desc="Processing files", unit="file"
    # ) as pbar:

    #     # Use ThreadPoolExecutor to run tasks in parallel
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         # Submit tasks to the executor
    #         future_to_file = {
    #             executor.submit(run_pipeline, *task): task[0]
    #             for task in get_tasks(files_to_process)
    #         }

    #         # Wait for tasks to complete and handle results/exceptions
    #         for future in concurrent.futures.as_completed(future_to_file):
    #             file_name = future_to_file[future]
    #             try:
    #                 future.result()  # Raise any exception caught during task execution
    #                 pbar.update(1)  # Update progress bar
    #                 # print(f"Successfully processed {file_name}") # Already printed in run_pipeline
    #             except Exception as exc:
    #                 print(f"{file_name} generated an exception: {exc}")

    for task in tqdm(
        get_tasks(files_to_process),
        total=len(files_to_process),
        desc="Processing files",
        unit="file",
    ):
        run_pipeline(*task)

    print("\nProcessing finished.")


if __name__ == "__main__":
    main()
