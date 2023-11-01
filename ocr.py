import argparse
import numpy as np
from PIL import Image as im, ImageFilter as imf, ImageEnhance as ime
import pytesseract
import os, glob, sys
import re, psutil
import cv2
from tqdm import tqdm
import datetime
import openslide as osi
import PIL.ImageDraw as ImageDraw
import multiprocessing as mp
import matplotlib.pyplot as plt
import PIL.ImageFont as ImageFont

font_fname = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
font_size = 30
font = ImageFont.truetype(font_fname, font_size)
DEFAULT_CONFIDENCE_THRESHOLD = 10
DEFAULT_DELIMITER = '+'
delimiter = DEFAULT_DELIMITER

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--slide_paths", help = "Specify directory of slides")
    parser.add_argument("-t", "--confidence_threshold", type=int, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Set confidence threshold of OCR (default = 80%)")
    parser.add_argument("-o", "--output_location", help="Path to the location of output csv file")
    parser.add_argument("-l", "--label_dir", help="Specify directory of processed labels")
    parser.add_argument("-w", "--num_workers", type=int,
                        help="Number of worker processes. "
                             "Default sets the number of worker processes to the number of CPU processes.")
    parser.add_argument("--delimiter", help="This character will be used to concat the OCR results"
                        , default=DEFAULT_DELIMITER)

    args = parser.parse_args()
    return args


def check_paths(args):
    global delimiter
    if args.slide_paths and os.path.exists(args.slide_paths):
        slide_paths = args.slide_paths
    else:
        raise ValueError(f'{args.slide_paths} does not exist or it is not accessible!')

    if args.confidence_threshold:
        if 0 <= args.confidence_threshold <= 100:
            confidence_threshold = args.confidence_threshold
        else:
            raise ValueError(f"confidence_threshold not in range [0 99]. f{args.confidence_threshold} is invalid")

    if args.output_location:
        output_file_location = args.output_location
        os.makedirs(output_file_location, exist_ok=True)
    else:
        raise ValueError(f'{args.output_file_location} does not exist or it is not accessible!')

    if args.label_dir:
        label_dir = args.label_dir
        os.makedirs(label_dir, exist_ok=True)
    else:
        raise ValueError('No label_dir given')

    if args.num_workers:
        n_process = args.num_workers
    else:
        n_process = psutil.cpu_count()

    delimiter = args.delimiter

    print("--------------------------------------------------")
    print("Running labelOCR with parameters:")
    print(f"Slide  Directory:  {slide_paths}")
    print(f"Label  Directory:  {label_dir}")
    print(f"Output File Path:  {output_file_location}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print("--------------------------------------------------")

    return slide_paths, label_dir, output_file_location, confidence_threshold, n_process

def get_slide_paths(rootpath, extensions=['tiff', 'tif', 'svs', 'scn']):
    """Get paths of slides that should be extracted.
    """
    paths = []
    for extension in extensions:
        path_wildcard = rootpath
        path_wildcard = os.path.join(path_wildcard, '*.' + extension)
        paths.extend(glob.glob(path_wildcard))
    return paths

def read_label(slide_path, label_dir, confidence_threshold, send_end):
    try:
        # Open image
        originalImage = osi.OpenSlide(slide_path)
        # Extract label
        labelImage = originalImage.associated_images['label']
        # Convert to grayscale
        bwLabelImage = labelImage.convert('L')
        # Rotate image to right orientation, changing image dimensions
        bwLabelImage = bwLabelImage.rotate(-90, expand=1)
        # Save Label
        data = pytesseract.image_to_data(bwLabelImage, output_type=pytesseract.Output.DICT, lang="Impact")
        ocr_result = ""
        for te, conf in zip(data['text'], data['conf']):
            te.replace("'", "")
            if len(te) > 1 and conf > confidence_threshold:
                ocr_result += te + delimiter

        draw = ImageDraw.Draw(bwLabelImage)
        w, h = font.getsize(ocr_result)
        draw.rectangle((0, 0, w, h), fill='white')
        draw.text((0, 0), ocr_result[:-1], font=font, fill='rgb(0, 0, 0)')
        if len(ocr_result)< 2 :
            ocr_result = os.path.split(slide_path)[-1]+" "
        ocr_result = os.path.basename(slide_path).split('.')[0] + "|~|~|" + " ".join(ocr_result.split()).replace('/','-')
        label = os.path.join(label_dir, ocr_result[:-1] + ".png")
        # if os.path.exists(label):
            # label = label[:-4] + os.path.basename(slide_path).split('.')[0] + ".png"
        bwLabelImage.save(label)
        send_end.send((slide_path, ocr_result))

    except Exception as e:
        print(f"could not process {slide_path}\n{e}")
        send_end.send((slide_path, ""))


# ---------------------------- Main function: ----------------------------
def main():

    # -----------------------
    # Set up argument parser:
    # -----------------------
    args = parse_input()
    slide_paths, label_dir, output_file_location, confidence_threshold, n_process = check_paths(args)

    str_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    os.makedirs(output_file_location, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    results_to_write = []
    slides = get_slide_paths(slide_paths)
    prefix = "Reading label from slides: "
    for idx in tqdm(range(0, len(slides), n_process), desc=prefix, dynamic_ncols=True):
        cur_slide_paths = slides[idx:idx + n_process]
        processes = []
        recv_end_list = []
        for path in cur_slide_paths:
            recv_end, send_end = mp.Pipe(False)
            recv_end_list.append(recv_end)
            p = mp.Process(target=read_label, args=(path, label_dir, confidence_threshold, send_end))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        results_to_write.extend(map(lambda x: x.recv(), recv_end_list))


    output_file_path = os.path.join(output_file_location, str_time + '_OCR.csv')
    output_file = open(output_file_path, "w")
    output_file.write("path,ocr_result\n")
    output_file.write("\n".join([f"{path},{ocr_result}" for path, ocr_result in results_to_write]))
    output_file.close()

    # End of function
    return

# ----------------------------------------------------------------------------

# Run main
if __name__ == '__main__':
    main()
