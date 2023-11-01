import argparse
import numpy as np
from PIL import Image as im, ImageFilter as imf, ImageEnhance as ime
import pytesseract
import os, glob, sys
import re
import cv2
import datetime
import openslide as osi
import matplotlib.pyplot as plt


# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)

output_file = None

def parse_input():

    parser = argparse.ArgumentParser()

    # Add option to change directory of images away from current working directory
    parser.add_argument("-d", "--dir_location", help = "Specify directory of files (default directory is current directory)")
    parser.add_argument("-t", "--thresh", type = int, help = "Set confidence threshold of OCR (default = 80)")
    parser.add_argument("--output_location", help = "local location of output file")
    parser.add_argument("--label_dir", help = "labels")

    args = parser.parse_args()

    return args


def check_paths(args):

    # Default confidence threshold is 80
    confThresh = 80

    # Assign imageDirPath as either default or configured
    if args.dir_location is not None:
        imageDirPath = args.dir_location
    else:
        raise ValueError('No dir_location given')

    # Assign confThresh as either default or configured
    if args.thresh is not None:
        # Make sure threshold is between 0 and 99
        if args.thresh in np.arange(100):
            confThresh = args.thresh
        # If not in appropriate range, terminate program with warning
        else:
            sys.exit("ERR: Argument -t, --thresh not in range [0 99]. Exiting...")

    if args.output_location != None:
        output_file_dir = args.output_location
    else:
        raise ValueError('No output_location given')

    if args.label_dir != None:
        label_dir = args.label_dir
    else:
        raise ValueError('No label_dir given')


    home  = os.path.expanduser('~')

    imageDirPath    = os.path.join(home, imageDirPath)
    label_dir       = os.path.join(home, label_dir)
    output_file_dir = os.path.join(home, output_file_dir)

    #Show info
    print("--------------------------------------------------")
    print("Running labelOCR with parameters:")
    print("Image  Directory:  " + imageDirPath)
    print("Label  Directory:  " + label_dir)
    print("Output Directory:  " + output_file_dir)
    print("Confidence Thresh: " + str(confThresh))
    print("--------------------------------------------------")

    return imageDirPath, output_file_dir, label_dir, confThresh


def labelocr_execute(imageDirPath, confThresh):

    # -------------
    # Locate files:
    # -------------

    # Create array fo all ".tiff" files in directory
    filesNameList = glob.glob(os.path.join(imageDirPath,"*.tiff"))

    # Check that array of files is not NULL
    if filesNameList == []:
        sys.exit("ERR: Directory " + imageDirPath + " contains no appropriate files. Exiting...")

    # Initialize counters
    fileCount = 0
    goodCount = 0
    handwrittenCount = 0
    weirdCount = 0
    lowConfCount = 0

    # -------- Loop over files --------
    for file in filesNameList:
        # In case of error (hard to parse edge cases, just iterate to next one)
        try:
            # Increment counter
            fileCount = fileCount + 1


            # Open image
            originalImage = osi.OpenSlide(file)

            # Extract label
            labelImage = originalImage.associated_images['label']

            # Convert to grayscale
            bwLabelImage = labelImage.convert('L')

            # Rotate image to right orientation, changing image dimensions
            bwLabelImage = bwLabelImage.rotate(90, expand = 1)

            # Save Label
            label_name = os.path.basename(file) + ".wrong.tiff"
            label      = os.path.join(label_dir, label_name)

            cv2.imwrite(label, np.array(bwLabelImage))

            # Imbinarize image using fixed threshold
            binaryLabelImage = bwLabelImage.point(lambda x: 0 if x < 180 else 255)

            # --------------------
            # Preliminary OCR run:
            # --------------------

            # Extract text from image
            data = pytesseract.image_to_data(binaryLabelImage, output_type = pytesseract.Output.DICT)

            # Initialize: "VOA" has not been found yet in first OCR run
            voaFound = 0

            # Initialize: good file name has not been found yet
            goodNameFound = 0

            # -------- Loop over extracted text snippets --------
            for k in range(np.size(data['text'])):

                # ---------------------
                # Form: "VOA-1234", "A"
                # ---------------------

                # Check for "VOA..." string that ends in number (extra A,B,C... not included)
                if "VOA" in data['text'][k] and \
                    re.search(r'\d+$', data['text'][k]) is not None:

                    # Flag that "VOA..." has been found
                    voaFound = 1

                    # ---------------------
                    # Secondary processing:
                    # ---------------------

                    # Crop in to include "VOA..." along with letter (assume next text snippet)
                    bwTextImage = bwLabelImage.crop(((np.min([data['left'][k], data['left'][k + 1]]) - 20), \
                                                     (np.min([data['top'][k], data['top'][k + 1]]) - 20), \
                                                    (np.max([data['left'][k] + data['width'][k], data['left'][k + 1] + data['width'][k + 1]]) + 20), \
                                                    (np.max([data['top'][k] + data['height'][k], data['top'][k + 1] + data['height'][k + 1]]) + 20)))

                    # Enhance contrast of image so that text gets crushed to black and slidee background is blown out to white
                    bwTextImage = ime.Contrast(bwTextImage).enhance(4)

                    # Blow out anything that is not text, i.e. grey artifact areas around the borders
                    binaryTextImage = bwTextImage.point(lambda x: 0 if x < 20 else 255)

                    # ---------------
                    # Second OCR run:
                    # ---------------

                    # Extract data from tile
                    refinedData = pytesseract.image_to_data(binaryTextImage, output_type = pytesseract.Output.DICT)

                    # print(refinedData)

                    # Initialize: "VOA" has not been found yet in second OCR run
                    voaFound2 = 0

                    # -------- Loop over extracted text snippets --------
                    for j in range(np.size(refinedData['text'])):

                        # Check for "VOA..." string that ends in number (extra A,B,C... not included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is not None:
                            #print("1a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that ends in letter (extra A,B,C...  included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is not None:
                            #print("2a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and refinedData['conf'][j] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term end in number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is not None:
                            #print("3a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                type(refinedData['conf'][j + 2]) != str and refinedData['conf'][j] > confThresh and \
                                refinedData['conf'][j + 1] > confThresh and refinedData['conf'][j + 2] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term ends in non-number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is None:
                            #print("4a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                    if voaFound2 == 0:
                        print("Trouble Parsing:" + data['text'][k] + data['text'][k + 1] + ", " + file)
                        weirdCount = weirdCount + 1

                    # Exit the loop over extracted text snippets
                    break

                # ----------------
                # Form: "VOA1234A"
                # ----------------

                # Check for "VOA..." string that doesn't end in number (e.g. the A, B, C already included)
                if "VOA" in data['text'][k] and \
                    re.search(r'\d+$', data['text'][k]) is None and \
                    re.search(r'[A-Z]$', data['text'][k]) is not None:
                    #print("2")
                    # Flag that "VOA..." has been voaFound
                    voaFound = 1

                    # ---------------------
                    # Secondary processing:
                    # ---------------------

                    # Crop in to include "VOA..." (letter already included)
                    bwTextImage = bwLabelImage.crop(((data['left'][k] - 20), (data['top'][k] - 20), (data['left'][k] + data['width'][k] + 20), (data['top'][k] + data['height'][k] + 20)))

                    # Enhance contrast of image so that text gets crushed to black and slide background is blown out to white
                    bwTextImage = ime.Contrast(bwTextImage).enhance(4)

                    # Blow out anything that is not text, i.e. grey artifact areas around the borders
                    binaryTextImage = bwTextImage.point(lambda x: 0 if x < 20 else 255)

                    # ---------------
                    # Second OCR run:
                    # ---------------

                    # Extract data from tile
                    refinedData = pytesseract.image_to_data(binaryTextImage, output_type = pytesseract.Output.DICT)

                    # Initialize: "VOA" has not been found yet in second OCR run
                    voaFound2 = 0

                    # -------- Loop over extracted text snippets --------
                    for j in range(np.size(refinedData['text'])):

                        # Check for "VOA..." string that ends in number (extra A,B,C... not included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is not None:
                            #print("1a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that ends in letter (extra A,B,C...  included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is not None:
                            #print("2a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and refinedData['conf'][j] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term end in number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is not None:
                            #print("3a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                type(refinedData['conf'][j + 2]) != str and refinedData['conf'][j] > confThresh and \
                                refinedData['conf'][j + 1] > confThresh and refinedData['conf'][j + 2] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term ends in non-number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is None:
                            #print("4a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                    if voaFound2 == 0:
                        print("Trouble Parsing:" + data['text'][k] + ", " + file)
                        weirdCount = weirdCount + 1

                    # Exit the loop over extracted text snippets
                    break

                # -------------------------
                # Form: "VOA-", "1234", "A"
                # -------------------------

                # Check for "VOA..." string that doesn't end in number or letter and second string ends in number
                if "VOA" in data['text'][k] and \
                    re.search(r'\d+$', data['text'][k]) is None and \
                    re.search(r'[A-Z]$', data['text'][k]) is None and \
                    re.search(r'\d+$', data['text'][k + 1]) is not None:
                    #print("3")
                    # Flag that "VOA..." has been found
                    voaFound = 1

                    # ---------------------
                    # Secondary processing:
                    # ---------------------

                    # Crop in to include "VOA..." along with number and letter (assume next two text snippet)
                    bwTextImage = bwLabelImage.crop(((np.min([data['left'][k], data['left'][k + 1], data['left'][k + 2]]) - 20), \
                                                     (np.min([data['top'][k], data['top'][k + 1], data['top'][k + 2]]) - 20), \
                                                    (np.max([data['left'][k] + data['width'][k], data['left'][k + 1] + data['width'][k + 1], data['left'][k + 2] + data['width'][k + 2]]) + 20), \
                                                    (np.max([data['top'][k] + data['height'][k], data['top'][k + 1] + data['height'][k + 1], data['top'][k + 2] + data['height'][k + 2]]) + 20)))

                    # Enhance contrast of image so that text gets crushed to black and slide background is blown out to white
                    bwTextImage = ime.Contrast(bwTextImage).enhance(4)

                    # Blow out anything that is not text, i.e. grey artifact areas around the borders
                    binaryTextImage = bwTextImage.point(lambda x: 0 if x < 20 else 255)

                    # ---------------
                    # Second OCR run:
                    # ---------------

                    # Extract data from tile
                    refinedData = pytesseract.image_to_data(binaryTextImage, output_type = pytesseract.Output.DICT)

                    # Initialize: "VOA" has not been found yet in second OCR run
                    voaFound2 = 0

                    # -------- Loop over extracted text snippets --------
                    for j in range(np.size(refinedData['text'])):

                        # Check for "VOA..." string that ends in number (extra A,B,C... not included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is not None:
                            #print("1a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that ends in letter (extra A,B,C...  included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is not None:
                            #print("2a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and refinedData['conf'][j] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term end in number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is not None:
                            #print("3a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                type(refinedData['conf'][j + 2]) != str and refinedData['conf'][j] > confThresh and \
                                refinedData['conf'][j + 1] > confThresh and refinedData['conf'][j + 2] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term ends in non-number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is None:
                            #print("4a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                    if voaFound2 == 0:
                        print("Trouble Parsing:" + data['text'][k] + data['text'][k + 1] + ", " + file)
                        weirdCount = weirdCount + 1

                    # Exit the loop over extracted text snippets
                    break

                # ----------------------
                # Form: "VOA-", "1234-A"
                # ----------------------

                # Check for "VOA..." string that doesn't end in number or letter and second string ends in non-number
                if "VOA" in data['text'][k] and \
                    re.search(r'\d+$', data['text'][k]) is None and \
                    re.search(r'[A-Z]$', data['text'][k]) is None and \
                    re.search(r'\d+$', data['text'][k + 1]) is None:
                    #print("4")
                    # Flag that "VOA..." has been found
                    voaFound = 1

                    # ---------------------
                    # Secondary processing:
                    # ---------------------

                    # Crop in to include "VOA..." along with number + letter (assume next text snippet)
                    bwTextImage = bwLabelImage.crop(((np.min([data['left'][k], data['left'][k + 1]]) - 20), \
                                                    (np.min([data['top'][k], data['top'][k + 1]]) - 20), \
                                                    (np.max([data['left'][k] + data['width'][k], data['left'][k + 1] + data['width'][k + 1]]) + 20), \
                                                    (np.max([data['top'][k] + data['height'][k], data['top'][k + 1] + data['height'][k + 1]]) + 20)))

                    # Enhance contrast of image so that text gets crushed to black and slide background is blown out to white
                    bwTextImage = ime.Contrast(bwTextImage).enhance(4)

                    # Blow out anything that is not text, i.e. grey artifact areas around the borders
                    binaryTextImage = bwTextImage.point(lambda x: 0 if x < 20 else 255)

                    # ---------------
                    # Second OCR run:
                    # ---------------

                    # Extract data from tile
                    refinedData = pytesseract.image_to_data(binaryTextImage, output_type = pytesseract.Output.DICT)

                    # Initialize: "VOA" has not been found yet in second OCR run
                    voaFound2 = 0

                    # -------- Loop over extracted text snippets --------
                    for j in range(np.size(refinedData['text'])):

                        # Check for "VOA..." string that ends in number (extra A,B,C... not included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is not None:
                            #print("1a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that ends in letter (extra A,B,C...  included)
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is not None:
                            #print("2a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and refinedData['conf'][j] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term end in number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is not None:
                            #print("3a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                type(refinedData['conf'][j + 2]) != str and refinedData['conf'][j] > confThresh and \
                                refinedData['conf'][j + 1] > confThresh and refinedData['conf'][j + 2] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + refinedData['text'][j + 2] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                        # Check for "VOA..." string that doesn't end in number or letter and second term ends in non-number
                        if "VOA" in refinedData['text'][j] and \
                            re.search(r'\d+$', refinedData['text'][j]) is None and \
                            re.search(r'[A-Z]$', refinedData['text'][j]) is None and \
                            re.search(r'\d+$', data['text'][k + 1]) is None:
                            #print("4a")
                            # Flag that "VOA..." has been found
                            voaFound2 = 1

                            # Ensure confidence is high enough
                            if type(refinedData['conf'][j]) != str and type(refinedData['conf'][j + 1]) != str and \
                                refinedData['conf'][j] > confThresh and refinedData['conf'][j + 1] > confThresh:

                                # Flag that good name has been found
                                goodNameFound = 1

                                # Export file name
                                fileNameComponents = refinedData['text'][j] + refinedData['text'][j + 1]
                                goodCount = goodCount + 1

                            else:

                                # Flag as low confidence
                                print("Low conf:" + refinedData['text'][j] + refinedData['text'][j + 1] + ", " + file)
                                lowConfCount = lowConfCount + 1

                            break

                    if voaFound2 == 0:
                        print("Trouble Parsing:" + data['text'][k] + data['text'][k + 1] + ", " + file)
                        weirdCount = weirdCount + 1


                    # Exit the loop over extracted text snippets
                    break

            # If we didn't find "VOA" in the extracted text, it is most likely handwritten
            if voaFound == 0:
                print("Handwritten:" + file)
                handwrittenCount = handwrittenCount + 1

            # Update stats every 50 slides
            if fileCount % 50 == 0:
                print("Total:       " + str(fileCount))
                print("Good:        " + str(goodCount) + "   " + str(goodCount/fileCount*100) + "%")
                print("Handwritten: " + str(handwrittenCount) + "   " + str(handwrittenCount/fileCount*100) + "%")
                print("Weird:       " + str(weirdCount) + "   " + str(weirdCount/fileCount*100) + "%")
                print("Low Conf:    " + str(lowConfCount) + "   " + str(lowConfCount/fileCount*100) + "%")

            # Normalize the format of filename
            if voaFound == 1 and voaFound2 == 1 and goodNameFound == 1:
                voaNumber = re.search('\d+', fileNameComponents).group(0)
                voaLetter = re.search('(?<![A-Z])[A-Z](?![A-Z])', fileNameComponents).group(0)[-1]

                legitFileName = "VOA-" + voaNumber + voaLetter

                confidence = refinedData['conf'][j]
                filename = file.replace(imageDirPath,"")
                output_file.write(legitFileName + "\t" + filename + "\t" + str(confidence) + "\n")

                tiff_name  = legitFileName + ".tiff"
                xml_name   = legitFileName + ".tiff.xml"
                label_name = legitFileName + ".tiff"

                tiff_path  = os.path.join(imageDirPath, tiff_name)
                xml_path   = os.path.join(imageDirPath, xml_name)
                label_path = os.path.join(label_dir, label_name)


                os.rename(file, tiff_path)
                os.rename(label, label_path)
                os.rename(file + ".xml", xml_path)


        #
        # If error occurs, goto next iteration
        except Exception as e:

            # Increment weirdCount and fileCount
            weirdCount = weirdCount + 1
            fileCount = fileCount + 1

            # Print output to acknowledge error
            print("ERR OCCURRED: " + file)
            print(e)
        #
            pass
    # Finished
    print("------------- FINISHED OPERATION -------------")
    print("Total:       " + str(fileCount))
    print("Good:        " + str(goodCount) + "   " + str(goodCount/fileCount*100) + "%")
    print("Handwritten: " + str(handwrittenCount) + "   " + str(handwrittenCount/fileCount*100) + "%")
    print("Weird:       " + str(weirdCount) + "   " + str(weirdCount/fileCount*100) + "%")
    print("Low Conf:    " + str(lowConfCount) + "   " + str(lowConfCount/fileCount*100) + "%")

    # End of function
    return



# ---------------------------- Main function: ----------------------------
def main():

    # ----------------------
    # Set default arguments:
    # ----------------------
    global output_file
    global label_dir

    # -----------------------
    # Set up argument parser:
    # -----------------------
    args = parse_input()

    imageDirPath, output_file_dir, label_dir, confThresh = check_paths(args)

    str_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    os.makedirs(output_file_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    output_file_path = os.path.join(output_file_dir, str_time + '_OCR.txt')

    output_file = open(output_file_path, "a")

    # Call labelocr_execute function
    labelocr_execute(imageDirPath, confThresh)

    output_file.close()

    # End of function
    return

# ----------------------------------------------------------------------------

# Run main
if __name__ == '__main__':
    main()
