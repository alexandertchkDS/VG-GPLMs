Pipeline Description
This document describes the pipeline, how to use the code, the machine learning model, how to improve it, and how to rewrite it better.

1. Creating Text Features
data_utils/create_txt_features.py

This script creates text features from a *.csv file to be in an appropriate view for the model.

2. Rename Video Files
data_utils/utils.py

This script renames video files.

3. Creating Video Features
data_utils/create_video_features.py

This script creates video features from *.mp4 files to be in an appropriate view for the model.

4. Splitting the Dataset
data_utils/split_dataset.py

This script splits the train.tok.txt and desc.tok.txt into train, dev, and test sets.

5. Setting Paths
scripts/T5_multimodal.sh

Set the necessary paths:

train_src_path
train_tgt_path
val_src_path
val_tgt_path
test_src_path
test_tgt_path
image_feature_path
Adjust the paths according to your file structure.

6. Training the Model
Run the following command:

scripts/T5_multimodal.sh

This will train the model.


Your markdown file looks good, but there are a few improvements and additions you can make:

Correct numbering: Your numbering is incorrect, two points are numbered as '2'.
Description: Add a brief description of what each step does and why it is necessary.
File Extensions: It's better to use backticks for file extensions like *.csv, *.mp4.
Paths: Mention that the paths need to be adjusted according to the user's file structure.
Shell Scripts: Mention that the shell scripts need to be made executable before running.
Requirements: List any requirements or prerequisites needed to run your code.
Formatting: Use code blocks for code, file paths, and commands.
Copy-Paste Friendly: Make sure the commands can be directly copied and pasted into the terminal.
Here's an improved version of your markdown file:

Pipeline Description
This document describes the pipeline, how to use the code, the machine learning model, how to improve it, and how to rewrite it better.

1. Creating Text Features
data_utils/create_txt_features.py

This script creates text features from a *.csv file to be in an appropriate view for the model.

2. Rename Video Files
data_utils/utils.py

This script renames video files.

3. Creating Video Features
data_utils/create_video_features.py

This script creates video features from *.mp4 files to be in an appropriate view for the model.

4. Splitting the Dataset
data_utils/split_dataset.py

This script splits the train.tok.txt and desc.tok.txt into train, dev, and test sets.

5. Setting Paths
scripts/T5_multimodal.sh

Set the necessary paths:

train_src_path
train_tgt_path
val_src_path
val_tgt_path
test_src_path
test_tgt_path
image_feature_path
Adjust the paths according to your file structure.

6. Training the Model
Run the following command:

arduino
Copy code
python ./src/run.py
This will train the model.

7. Testing the Model
data_utils/test_Bart_multimodal.sh

This script is used for testing.

Make sure to make the shell scripts executable before running them:
chmod +x scripts/T5_multimodal.sh
chmod +x data_utils/test_Bart_multimodal.sh

Requirements
Python 3.8
Oother requirements (use requirements.txt)
Make sure to install all the necessary requirements before running the code.

