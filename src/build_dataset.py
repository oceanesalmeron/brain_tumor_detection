import os

root = ".\data"

for path, subdirs, files in os.walk(root):
    for name in files:
        print('cat:', os.path.basename(path), 'file:',os.path.join(path, name))
        

# imagePaths = list(paths.list_images("../input/brain-mri-images-for-brain-tumor-detection"))
# data = []
# labels = []

# for imagePath in imagePaths:

# 	label = imagePath.split(os.path.sep)[-2]


# 	image = cv2.imread(imagePath)
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 	image = cv2.resize(image, (224, 224))


# 	data.append(image)
# 	labels.append(label)
