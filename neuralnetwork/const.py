import os

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(CURRENT_PATH)

SCUT_PATH = "SCUT-FBP5500_v2"
DATASET_PATH = os.path.join(CURRENT_PATH, "dataset", SCUT_PATH)
DATA_PATH = os.path.join(DATASET_PATH, "Images")
RATING_PATH = os.path.join(DATASET_PATH, "train_test_files", "All_labels.txt")
URATING_PATH = os.path.join(DATASET_PATH, "train_test_files", "User_labels.txt")
MODEL_PATH = os.path.join(PARENT_PATH, "common",
                          "haarcascade_frontalface_alt.xml")

DATA_URL = "https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
ZFILE = "SCUT-FBP5500_v2.1.zip"

DAT_PATH = os.path.join(CURRENT_PATH, "data")

TRAINING_FILE = os.path.join(DAT_PATH, "train_label_dist.dat")
TESTING_FILE = os.path.join(DAT_PATH, "test_label_dist.dat")

#TRAINEDMODEL_FILE = os.path.join(PARENT_PATH, "common", "beauty-resnet.h5")
TRAINEDMODEL_FILE = os.path.join(PARENT_PATH, "common", "model-ldl-resnet.h5")
