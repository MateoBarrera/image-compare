# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from image_similarity_measures.evaluate import evaluation
from skimage.metrics import *


def get_name_image(image: str):
    return image.split("_")[-1].split(".")[0]


def load_image(path_name, type_img=cv2.IMREAD_GRAYSCALE):
    return cv2.imread(path_name, type_img)


def evaluate(image1, image2):
    """

    :return:
    """
    im_name1 = get_name_image(image1)
    im_name2 = get_name_image(image2)
    print('Evaluation start')
    print(f'Comparing {im_name1} vs {im_name2}')
    print(evaluation(org_img_path=image1,
                     pred_img_path=image2,
                     metrics=["issm"]))
    print("--------------------------")
    # metrics = ["rmse", "psnr", "ssim", "fsim", "issm", "sre", "sam", "uiq"]


def ssim_skit(image1, image2):
    im_name1 = get_name_image(image1)
    im_name2 = get_name_image(image2)
    print('Evaluation start')
    print(f'Comparing {im_name1} vs {im_name2}')

    print(f"'nssim': {structural_similarity(load_image(image1), load_image(image2))}")


def nmi_skit(image1, image2):
    im_name1 = get_name_image(image1)
    im_name2 = get_name_image(image2)
    print('Evaluation start')
    print(f'Comparing {im_name1} vs {im_name2}')

    print(f"'nmi': {normalized_mutual_information(load_image(image1), load_image(image2))}")


def nmrse_skit(image1, image2):
    im_name1 = get_name_image(image1)
    im_name2 = get_name_image(image2)
    print('Evaluation start')
    print(f'Comparing {im_name1} vs {im_name2}')

    print(f"'nmrse': {normalized_root_mse(load_image(image1), load_image(image2))}")


def vi_skit(image1, image2):
    im_name1 = get_name_image(image1)
    im_name2 = get_name_image(image2)
    print('Evaluation start')
    print(f'Comparing {im_name1} vs {im_name2}')

    print(f"'vi': {variation_of_information(load_image(image1), load_image(image2))}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image1_path = "src/Mosaico-DataSet1_DroneFacade.jpg"
    image2_path = "src/Mosaico-DataSet1_ImageCompositejpg.jpg"
    # evaluate(image1_path, image2_path)
    ssim_skit(image1_path, image2_path)

    image1_path = "src/Mosaico-DataSet1_DroneFacade.jpg"
    image2_path = "src/Mosaico-DataSet1_PTGui.jpg"
    # evaluate(image1_path, image2_path)
    ssim_skit(image1_path, image2_path)
    #
    image1_path = "src/Mosaico-DataSet1_PTGui.jpg"
    image2_path = "src/Mosaico-DataSet1_ImageCompositejpg.jpg"
    # evaluate(image1_path, image2_path)
    ssim_skit(image1_path, image2_path)
