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
## Arguments
Here's an explanation of arguments:

  - `dir_location` : Path to the directory of slides that we want to extract their labels
  - `thresh` : Confidence threshold of OCR (default = 80)
  - `output_location` : Path to save the output log
  - `label_dir` : Path to save the labels


## Usage

The OCR code extract the label of each WSI, and rename that slide based on the label's name. 

Note that the code is not able to detect the label from the image, so I have made the `label_dir` argument to save the labels in a folder. Then you have to **Manually** rename those files.

*Important*: Always check the predicted label with the correspondece label image to make sure that the code has predicted correctly.   

