# OCR

## Singularity
To build the singularity image do:

```
singularity build --remote singularity_ocr.sif Singularity.def
```

In the SH file, you should bind the path to the slides if the slides is not in the singularity working directory.

```
singularity run -B /projects/ovcare/classification/WSI singularity_ocr.sif --dir_location path/to/slides --output_location path/to/output --label_dir path/to/save/labels
```
If you are running this code on your local machine you need to run the following command to copy the train model to the tesseract directory:
```
sudo cp train/output/Impact.traineddata /usr/share/tesseract-ocr/4.00/tessdata/xyz.traineddata
```
## Arguments
Here's an explanation of arguments:

  - `slide_paths` : Path to the directory of slides that we want to extract their labels
  - `confidence_threshold` : Confidence threshold of OCR (default = 80)
  - `output_location` : Path to the location of output csv file
  - `label_dir` : Specify directory of processed labels
  - `num_workers` : Number of worker processes. Default sets the number of worker processes to the number of CPU processes.


## Usage

The OCR code extract the label of each WSI, and rename that slide based on the label's name. 

You can use the train folder, and the script named "tesstrainDone.sh" to train a new model to detect new set of fonts.

*Important*: Always check the predicted label with the correspondece label image to make sure that the code has predicted correctly.   

