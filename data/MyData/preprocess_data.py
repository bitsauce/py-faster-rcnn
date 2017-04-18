import os
import pandas as pd

# Read object-dataset CSV file
object_dataset = pd.read_csv("labels.csv", sep=' ')

# Write all image files to train.txt
with open("data/ImageSets/train.txt", 'w') as f:
	files = os.listdir("data/Images/")
	num_files = len(files)
	for i, file_name in enumerate(files):
		print("Processing file %i/%i" % (i + 1, num_files))

		# Create annotations file using pd.DataFrame.to_csv
		objects = object_dataset[object_dataset.frame == file_name]
		del objects["frame"]
		objects.to_csv("data/Annotations/%s.csv" % file_name[:-4], index=False)

		# Write to train.txt
		f.write("%s\n" % (file_name[:-4]))