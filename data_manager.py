import math
import os
import shutil
import tarfile


# import constants as c


class DataManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def extract_dataset(self, compressed_dataset_file_name, dataset_directory):
        try:
            # extract files to dataset folder
            tar = tarfile.open(compressed_dataset_file_name, "r:gz")
            tar.extractall(dataset_directory)
            tar.close()
            print("Files extraction was successfull ...")

        except:
            print("Exception raised: No extraction was done ...")

    def make_folder(self, folder_path):
        try:
            os.mkdir(folder_path)
            print(folder_path, "was created ...")
        except:
            print("Exception raised: ", folder_path, "could not be created ...")

    def move_files(self, src, dst, group):
        for fname in group:
            shutil.copy(src + '/' + fname, dst + '/' + fname)

    def get_fnames_from_dict(self, dataset_dict, key):
        training_data, testing_data = [], []

        # get list lengths
        length_data = len(dataset_dict[key])
        length_separator = math.trunc(length_data * 2 / 3)

        # build traing and testing lists
        training_data += dataset_dict[key][:length_separator]
        testing_data += dataset_dict[key][length_separator:]

        return training_data, testing_data

    def manage(self):

        TRAINING_PATH = "TrainingDataVCTK"
        TESTING_PATH = "TestingDataVCTK"

        # read config file and get path to compressed dataset
        compressed_dataset_file_name = self.dataset_path
        dataset_directory = compressed_dataset_file_name.split(".")[0]
        print(dataset_directory)


        # create a folder for the data
        try:
            os.mkdir(dataset_directory)
        except:
            pass
        #
        # # extract dataset
        # self.extract_dataset(compressed_dataset_file_name, dataset_directory)

        # select females files and males files
        speaker_names = [fname for fname in os.listdir(dataset_directory)]
        dataset_dict = {}
        # {"f0001": [], "f0002": [], "f0003": [], "f0004": [], "f0005": [],
        #                 "m0001": [], "m0002": [], "m0003": [], "m0004": [], "m0005": [], }

        for name in speaker_names:
            directory = dataset_directory + "/" + name
            # file_list = os.listdir(directory)
            dataset_dict[name] = os.listdir(directory)

        # print(dataset_dict)

        # # fill in dictionary+
        # for fname in speaker_names:
        #     dataset_dict[fname.split('_')[0]].append(fname)

        # divide and group file names
        training_set, testing_set = {}, {}

        # make training and testing foldersget_fnames_from_dict
        self.make_folder(TRAINING_PATH)
        self.make_folder(TESTING_PATH)

        # make folders with speakers ids and move data
        for key in dataset_dict.keys():
            # separate data into training and testing data
            training_set[key], testing_set[key] = self.get_fnames_from_dict(dataset_dict, key)

            # make folders
            self.make_folder(TRAINING_PATH + "/" + key)
            self.make_folder(TESTING_PATH + "/" + key)

            # move files
            self.move_files(dataset_directory + "/" + key, TRAINING_PATH + "/" + key, training_set[key])
            self.move_files(dataset_directory + "/" + key, TESTING_PATH + "/" + key, testing_set[key])

        # print('training set : ', len(training_set.keys()))
        # print('test set : ', len(testing_set.keys()))

if __name__ == "__main__":
    # data_manager = DataManager("/home/hindu/amicorpus_data/dataset")
    data_manager = DataManager("/home/hindu/VCTKcorpus_data/data-grouped")
    data_manager.manage()
