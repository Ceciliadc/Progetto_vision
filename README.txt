This is the structure of our project:

- union_final
	- audio_description
	- descriptions
	- interface		(saves the video with bounding boxes around paintings resulting from yolo and all the labels)
	- people_detection	(outputs a video with bounding boxes around people)
	- pose_estimation
	- rectification
	- retrieval
	- sift			(contains also the image database and csv file)
	- yolov5
	- requirements.txt

IMPORTANT NOTE: In order to correctly perform the whole pipeline it is necessary to run the program on a GPU; the performances are related to it. 

Open the unzipped project folder with PyCharm. To execute the program, you need to create a new virtual environment with a conda interpreter and install all the indicated requirements. In the Run/Debug configurations choose as Script Path "union_final/interface/interface.py" and then you just have to run the code. 

The first interface page will appear where you have to select the video you want to process and then, after the program have performed all the steps (it could take some minutes depending on the lenght of the video and the GPU used), a second interface page will appear with all the rectified and found paintings. Choosing a particular painting will open a final page with all the information regarding it, including an audio description that can be played with the play button.

