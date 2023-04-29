import cv2
import glob
import imutils
import logging
import json
import numpy as np
import os
import pandas as pd
import pickle
import requests
import shutil
import time as t
import urllib.request
from PIL import Image
from datetime import datetime
from flask import Flask, request
from utils.range_model import *
from utils.size_uniformity_model import *
from utils.watershed_size_model import *
from utils.imgpro_queries import *
from utils.rescale import *
from utils.config import *

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')

app = Flask(__name__)


def check_configurations():
    """
    Function to check the Configurations...
    :return: Return Boolean True or False.
    """

    logging.info("Checking configurations...")
    try:
        logging.info("Configurations are correct...!!!")
        return True
    except Exception as e:
        logging.error(f"Configurations Error...{e}")
        return False


def download_image(ticket_id, image_id, url_image_path, server):
    """
    Definition to download the input image from the image server using image_url
    UAT - https://igrade-testing.waycool.in
    LIVE - https://igrade.waycool.in
    :param ticket_id: ticket_id of the image which needs to download
    :param image_id: Image_id of the image
    :param url_image_path: URL link of image to download
    :param server: Server name either 'uat' or 'live'
    :return: Return the path in server where the image is downloaded
    """

    url_to_download = image_server[server] + str(url_image_path)
    logging.info(f"url_to_download : {url_to_download}")
    path_to_download = input_images_path + str(ticket_id) + '_' + str(image_id) + ".jpg"
    logging.info(f"path_to_download : {path_to_download}")
    urllib.request.urlretrieve(url_to_download, path_to_download)
    logging.info(f'Image downloaded!!!')
    return path_to_download


def size_model(ticket_id, image_id, image_path):
    """
    Function to calculate the detected object size like length and width.
    :param ticket_id: Ticket id of the request.
    :param image_id: Image id of the request.
    :param image_path: Image path of the request.
    :return: Return the list of length and width details.
    """

    try:
        logging.info("Size model started...")
        data, length_sizes_list, width_sizes_list = size_detection(ticket_id, image_id, image_path)
        return 3, data, length_sizes_list, width_sizes_list

    except Exception as e:
        print(e)
        print('[Error] : in size model')
        return 4, [], [], []


def touched_size_model(ticket_id, image_id, image_path, commodity_id, coin_diameter):
    """
    Function to call the watershed model based on the commodity.
    :param ticket_id: Ticket id of the request.
    :param image_id: Image id of the request.
    :param image_path: Image path of the request.
    :param commodity_id: Commodity id of the request.
    :param coin_diameter: Coin Diameter entered by the user.
    :return: Return the object detection results of the commmodity.
    """

    try:
        logging.info("Watershed model started...")
        status = ''
        data = []
        length_sizes_list = []
        width_sizes_list = []
        if commodity_id in black_bg_map:
            status, data, length_sizes_list, width_sizes_list = black_bg_size_detection(ticket_id, image_id, image_path,
                                                                                        coin_diameter)
        elif commodity_id in other_small_comm_map:
            status, data, length_sizes_list, width_sizes_list = other_small_comm_size_detection(ticket_id, image_id,
                                                                                                image_path,
                                                                                                coin_diameter)
        elif commodity_id in gka_bg_map:
            status, data, length_sizes_list, width_sizes_list = gka_bg_size_detection(ticket_id, image_id, image_path,
                                                                                      coin_diameter)
        elif commodity_id in white_bg_map:
            status, data, length_sizes_list, width_sizes_list = white_bg_size_detection(ticket_id, image_id, image_path,
                                                                                        coin_diameter)

        elif commodity_id in watershed_model_map.keys():
            if watershed_model_map[commodity_id] == 'wheat_bg':
                status, data, length_sizes_list, width_sizes_list = wheat_bg_size_detection(ticket_id,
                                                                                            image_id,
                                                                                            image_path,
                                                                                            coin_diameter)
            elif watershed_model_map[commodity_id] == 'maize_bg':
                status, data, length_sizes_list, width_sizes_list = maize_bg_size_detection(ticket_id,
                                                                                            image_id,
                                                                                            image_path,
                                                                                            coin_diameter)
            elif watershed_model_map[commodity_id] == 'rice_bg':
                status, data, length_sizes_list, width_sizes_list = rice_bg_size_detection(ticket_id,
                                                                                           image_id,
                                                                                           image_path,
                                                                                           coin_diameter)
            elif watershed_model_map[commodity_id] == 'green_gram_bg':
                status, data, length_sizes_list, width_sizes_list = green_gram_bg_size_detection(ticket_id,
                                                                                                 image_id,
                                                                                                 image_path,
                                                                                                 coin_diameter)
            elif watershed_model_map[commodity_id] == 'green_gram_splits_bg':
                status, data, length_sizes_list, width_sizes_list = green_gram_splits_bg_size_detection(ticket_id,
                                                                                                        image_id,
                                                                                                        image_path,
                                                                                                        coin_diameter)
            elif watershed_model_map[commodity_id] == 'sesame_seeds_wg':
                status, data, length_sizes_list, width_sizes_list = sesame_seeds_wg_size_detection(ticket_id,
                                                                                                   image_id,
                                                                                                   image_path,
                                                                                                   coin_diameter)
            elif watershed_model_map[commodity_id] == 'niger_seeds_wg':
                status, data, length_sizes_list, width_sizes_list = niger_seeds_wg_size_detection(ticket_id,
                                                                                                  image_id,
                                                                                                  image_path,
                                                                                                  coin_diameter)
            elif watershed_model_map[commodity_id] == 'black_pepper_wg':
                status, data, length_sizes_list, width_sizes_list = black_pepper_wg_size_detection(ticket_id,
                                                                                                   image_id,
                                                                                                   image_path,
                                                                                                   coin_diameter)
            elif watershed_model_map[commodity_id] == 'kidney_beans_wg':
                status, data, length_sizes_list, width_sizes_list = kidney_beans_wg_size_detection(ticket_id,
                                                                                                   image_id,
                                                                                                   image_path,
                                                                                                   coin_diameter)
            elif watershed_model_map[commodity_id] == 'ragi_wg':
                status, data, length_sizes_list, width_sizes_list = ragi_wg_size_detection(ticket_id,
                                                                                           image_id,
                                                                                           image_path,
                                                                                           coin_diameter)
            elif watershed_model_map[commodity_id] == 'red_gram_white_bg':
                status, data, length_sizes_list, width_sizes_list = red_gram_white_bg_size_detection(ticket_id,
                                                                                                     image_id,
                                                                                                     image_path,
                                                                                                     coin_diameter)
            elif watershed_model_map[commodity_id] == 'cardamom':
                status, data, length_sizes_list, width_sizes_list = cardamom_size_detection(ticket_id,
                                                                                            image_id,
                                                                                            image_path,
                                                                                            coin_diameter)
            elif watershed_model_map[commodity_id] == 'pista':
                status, data, length_sizes_list, width_sizes_list = pista_size_detection(ticket_id,
                                                                                         image_id,
                                                                                         image_path,
                                                                                         coin_diameter)
            elif watershed_model_map[commodity_id] == 'raisin':
                status, data, length_sizes_list, width_sizes_list = raisin_size_detection(ticket_id,
                                                                                          image_id,
                                                                                          image_path,
                                                                                          coin_diameter)

            else:
                logging.info("No matching str name:  check dict value and if condition value")
        else:
            logging.info("Requested commodity is not added.. for touched grains prediction")
            # running black_bg as default model
            status, data, length_sizes_list, width_sizes_list = black_bg_size_detection(ticket_id,
                                                                                        image_id,
                                                                                        image_path,
                                                                                        coin_diameter)
        return status, data, length_sizes_list, width_sizes_list

    except Exception as e:
        logging.error(f'Error in size model {e}')
        return 4, [], [], []


def range_model(length_sizes_list):
    """
    Function to calculate the range of object size.
    :param length_sizes_list: List of objects length to calc size range.
    :return: Return the dict of range details.
    """

    try:
        logging.info("Range model started...")
        data = range_calculation(length_sizes_list)
        return 3, data

    except Exception as e:
        logging.error(f'Error in range model {e}')
        return 4, []


def size_uniformity_model(sizes_list):
    """
    Function to uniformize the size of detected objects.
    :param sizes_list: List of length and width of detected objects.
    :return: Return the dict of uniformized size details.
    """

    try:
        logging.info("Size Uniformity model started...")
        data = size_uniformity_master_function(sizes_list)
        return 3, data
    except Exception as e:
        logging.error(f'Error in size uniformity model {e}')
        return 4, {}


def club_range(range_data):
    """
    Function to club the objects size ranges.
    :param range_data: Dictionary of range data.
    :return: Return the clubbed range data as dict.
    """

    checker = 0
    clubbed_percent = 0.0
    range_ = '0.0 - '
    count_ = 0
    exists = False
    while range_data[0]['range_percentage'] < range_clubbing_percent:
        exists = True
        if range_data[0]['range_percentage'] == 0.0 and checker == 0:
            range_ = range_data[0]['range'].split("-")[-1][1:] + ' - '
            del range_data[0]

        else:
            checker = 1
            clubbed_percent += range_data[0]['range_percentage']
            count_ += range_data[0]['count']
            del range_data[0]
        if not range_data:
            return [{'range': '0.0 - 20.0', 'count': 0, 'range_percentage': 0.0}]

    if exists:
        range_ = range_ + range_data[0]['range'].split("-")[0][:-1]
        dic = {'range': range_, 'count': count_, 'range_percentage': clubbed_percent}
        range_data.insert(0, dic)

    checker = 0
    clubbed_percent = 0.0
    range_ = ' - 30.0'
    count_ = 0
    exists = False
    while range_data[-1]['range_percentage'] < range_clubbing_percent:
        exists = True
        if range_data[-1]['range_percentage'] == 0.0 and checker == 0:
            del range_data[-1]

        else:
            checker = 1
            clubbed_percent += range_data[-1]['range_percentage']
            count_ += range_data[-1]['count']
            range_ = ' - ' + range_data[-1]['range'].split("-")[-1][1:]
            del range_data[-1]
    if exists:
        range_ = range_data[-1]['range'].split("-")[-1][1:] + range_
        dic = {'range': range_, 'count': count_, 'range_percentage': clubbed_percent}
        range_data.append(dic)
    return range_data


def get_edit_grade_details(length_size_list, grade_details):
    """
    Function to get the and edit the grade details.
    :param length_size_list: List of length size.
    :param grade_details: List of grade details.
    :return: Return full data and bold count and others count.
    """

    try:
        logging.info("Get_edit_grade_details started...")
        data, bold_count, others_count = grade_range(length_size_list, grade_details)
        print('bold_count', bold_count)
        return 3, data, bold_count, others_count

    except Exception as e:
        logging.error(f'Error in get_edit_grade_details model - {e}')
        return 4, None, None, None


def get_overall_size_details(ticket_id):
    """
    Function to calculate the overall size data
    :param ticket_id: Ticket id of the request.
    :return: Return the overall data, length and width list.
    """

    try:
        logging.info("Get_overall_size_details started...")
        rows = get_combined_overall_details(ticket_id)

        if not rows:
            logging.error("Error Requested Ticket - Id not found...")
            return 4, [], [], {}

        rows = rows[0]
        combined_overall_details = rows[1]

        combined_overall_details = json.loads(combined_overall_details)
        overall_size_data = combined_overall_details['overall_size_data']
        length_list = []
        width_list = []
        for i in overall_size_data:
            length_list.append(float(i['length']))
            width_list.append(float(i['width']))

        return 3, length_list, width_list, combined_overall_details

    except Exception as e:
        logging.error(f'Error in get_overall_size_details model {e}')
        return 4, [], [], {}


def get_grade_details(length_size_list, commodity_id):
    """
    Function to calculate the grade details.
    :param length_size_list: List of length size.
    :param commodity_id: Commodity id of the request.
    :return: Return the final data.
    """

    grade_details = {}
    row = get_commodity_range_grade_details(commodity_id)

    if row == [] or None:
        logging.error("Error in the getting grade details from DB...")
        return 4, None, None, None
    elif len(row) > 1:
        logging.error("More than one results...")
        return 4, None, None, None

    row = row[0]
    grade_details["small_s"], grade_details["small_t"], grade_details["regular_s"], grade_details["regular_t"], \
        grade_details["bold_s"], grade_details["bold_t"] = str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(
        row[4]), str(row[5])

    for i in grade_details.values():
        if i == 'None' or i == '':
            logging.info("Invalid grade range data in DataBase...")
            return 4, None, None, None

    try:
        logging.info("Range Grade estimation started...")
        data, bold_count, others_count = grade_range(length_size_list, grade_details)
        return 3, data, bold_count, others_count

    except Exception as e:
        logging.error(f'Error in size uniformity model - {e}')
        return 4, None, None, None


# def check_anomaly(image_path):
#     """
#     Function to check the anomaly image in request.
#     :param image_path: Image path of the request.
#     :return: Return the boolean value.
#     """
#
#     image = cv2.imread(image_path)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     features = quantify_image(hsv, bins=(3, 3, 3))
#     preds = anomaly_model.predict([features])[0]
#     if preds == -1:
#         return True  # It is Anomaly
#     else:
#         return False


def find_width_min_max_avg(overall_size_data):
    """
    Function to find the maximum and minimum width.
    :param overall_size_data: Dict of overall size data.
    :return: Return the maximum and minimum width.
    """

    if overall_size_data == []:
        return 0.0, 0.0, 0.0
    mini = 10000.0
    maxi = 0.0
    summ = 0.0
    for i in overall_size_data:
        wid = float(i['width'])
        summ = summ + wid
        if wid > maxi:
            maxi = wid
        if wid < mini:
            mini = wid

    aver = summ / len(overall_size_data)
    print("WIDTH :   min-", mini, '     maxi-', maxi, '    average-', aver)
    return maxi, mini, aver


def find_length_min_max_avg(overall_size_data):
    """
    Function to find the maximum and minimum length.
    :param overall_size_data: Dict of overall size data.
    :return: Return the maximum and minimum length.
    """

    if overall_size_data == []:
        return 0.0, 0.0, 0.0
    mini = 10000.0
    maxi = 0.0
    summ = 0.0
    for i in overall_size_data:
        leng = float(i['length'])
        summ = summ + leng
        if leng > maxi:
            maxi = leng
        if leng < mini:
            mini = leng

    aver = summ / len(overall_size_data)
    print("LENGTH :  min-", mini, '     maxi-', maxi, '    average-', aver)
    return maxi, mini, aver


def upload_output_image(ticket_id, image_name_list):
    """
    Function to upload the output image to the image server.
    :param ticket_id: Ticket ID of the request
    :param image_name_list: List of output images.
    """
    try:
        output_folder_to_zip = output_images_path + str(ticket_id)

        if not os.path.exists(output_folder_to_zip):
            os.mkdir(output_folder_to_zip)

        for image_name in image_name_list:
            shutil.copy(output_images_path + image_name, output_folder_to_zip)

        shutil.make_archive(output_folder_to_zip,
                            'zip',
                            root_dir=output_images_path)

        images_zip = open(output_folder_to_zip + '.zip', 'rb')
        files = {"output_images": images_zip}
        data = {"request_id": ticket_id}
        getdata = requests.post(upload_image_url, data=data, files=files)

        data = json.loads(getdata.text)
        images_zip.close()
        if data['status']:
            shutil.rmtree(output_folder_to_zip)
            os.remove(output_folder_to_zip + '.zip')
            for image_name in image_name_list:
                os.remove(output_images_path + image_name)

    except Exception as e:
        logging.error(f'{e}')


@app.route('/v1.0_edit_range_grade', methods=['GET', 'POST'])
def edit_range_grade():
    if request.method == 'POST':
        ticket_id = request.form['ticket_id']
        user_id = request.form['user_id']
        server = request.form['server']
        regular_from = float(request.form['regular_from'])
        regular_to = float(request.form['regular_to'])

        if server != 'uat':
            return {"message": "Hitted UAT server, but request server key-value is not 'uat'", "status": "0"}

        # global mydb
        mydb, connected = db_connect(server)

        status, length_list, width_list, combined_overall_data = get_overall_size_details(ticket_id)

        grade_details = {'small_s': start_range, 'small_t': regular_from, 'regular_s': regular_from,
                         'regular_t': regular_to, 'bold_s': regular_to, 'bold_t': end_range}

        if status == 3:
            status, grade_range_data, bold_count, others_count = get_edit_grade_details(length_list, grade_details)
        else:
            return {"message": "[Error] : in get_overall_size_details for the ticket-id : " + str(ticket_id),
                    "status": "1"}

        if status == 3:
            combined_overall_data["overall_range_grade_details"] = grade_range_data
            combined_overall_data["bold_ratio"] = "Bold : others  -  " + str(bold_count) + " : " + str(others_count)

            combined_overall_data = json.dumps(combined_overall_data, indent=1)
            combined_overall_data = json.loads(combined_overall_data)
            combined_overall_data = str(combined_overall_data).replace("'", '"')

            update_edit_grade_range_data(ticket_id, combined_overall_data)
            logging.info("DB connection closed...!!!")
            mydb.close()

            logging.info('Successfully edited range_grade details')
        else:
            return {"message": "[Error] : in  get_edit_grade_details for the ticket-id : " + str(ticket_id),
                    "status": "1"}

        return {"message": "Edit range grade done for the ticket-id : " + str(ticket_id), "status": "1"}


@app.route('/quick_imgpro/v1.0_quick_image_process', methods=['GET', 'POST'])
def size_range():
    """
    API to detect the objects and calculate the size of objects.
    :return: Return the dict of final output data.
    """

    if request.method == 'POST':
        start_time = t.time()
        ticket_id = request.form['ticket_id']
        user_id = request.form['user_id']
        server = request.form['server']
        if server != 'uat':
            return {"message": "Hitted UAT server, but request server key-value is not 'uat'", "status": "0"}

        # global mydb
        mydb, connected = db_connect(server)
        rows = get_image_details_quick_imgpro(ticket_id)

        if not rows:
            logging.info(f'No images present for the ticket id: {ticket_id}')
            return {"message": "No Images for the ticket-id : " + str(ticket_id), "status": "0"}

        commodity_id = get_commodity_id(ticket_id)[0]
        commodity_id = str(commodity_id[0])

        ticketid_data = {}
        overall_length_sizes_list = []
        overall_width_sizes_list = []
        overall_size_data = []
        overall_range_data = []
        overall_range_data_dict = {}

        for row in rows:
            image_id = str(row[0])
            url_image_path = str(row[1])
            image_weight = float(row[2])
            img_data = {}
            try:
                # Download pending image and saving in input_images folder
                downloaded_image_path = download_image(ticket_id, image_id, url_image_path, server)

            except Exception as e:
                logging.info(f'Error in downloading the image..'
                             f' \nThis is Production server.. may be live code will execute'
                             f'Error - {e}')
                continue

            # size calculation part
            img_data["size_status"], img_data[
                "size_data"], length_sizes_list, width_sizes_list = size_model(ticket_id,
                                                                               image_id,
                                                                               downloaded_image_path)
            img_data["size_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            length_sizes_list.sort()
            width_sizes_list.sort()

            if img_data["size_status"] == 3:
                overall_length_sizes_list = overall_length_sizes_list + length_sizes_list
                overall_size_data = overall_size_data + img_data["size_data"]

                overall_width_sizes_list = overall_width_sizes_list + width_sizes_list

                img_data["size_uniformity_status"], img_data["size_uniformity_data"] = size_uniformity_model(
                    length_sizes_list)
                img_data["size_uniformity_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Range calculation part..
                img_data["range_status"], img_data["range_data"] = range_model(length_sizes_list)
                img_data["range_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                img_data["image_total_weight"] = image_weight
                img_data["image_total_count"] = len(length_sizes_list)

                status, grade_range_data, bold_count, others_count = get_grade_details(width_sizes_list, commodity_id)

                if status == 3:
                    img_data["image_range_grade_details"] = grade_range_data
                    img_data["bold_ratio"] = "Bold : others  -  " + str(bold_count) + " : " + str(others_count)

                img_data["image_max_length"], img_data["image_min_length"], img_data[
                    "image_avg_lenght"] = find_length_min_max_avg(img_data["size_data"])
                img_data["image_max_width"], img_data["image_min_width"], img_data[
                    "image_avg_width"] = find_width_min_max_avg(img_data["size_data"])

                img_data["image_unclubbed_range_data"] = img_data["range_data"]

                if img_data["range_data"]:
                    img_data["range_data"] = club_range(img_data["range_data"])
                else:
                    img_data["range_data"] = []

                ticketid_data[image_id] = img_data

                if img_data["range_status"] == 3:
                    for r in img_data["range_data"]:
                        range_ = r["range"]
                        count = r["count"]
                        overall_range_data_dict[range_] = overall_range_data_dict.get(range_, 0) + count

            else:
                logging.error(f"Image Error for the Ticket_Image_id : {ticket_id} - {image_id}")

        ticketid_data = json.dumps(ticketid_data, indent=1)
        ticketid_data = json.loads(ticketid_data)
        ticketid_data = str(ticketid_data).replace("'", '"')

        logging.info("Combined overall data started...")

        combined_overall_data = {"overall_size_data": overall_size_data}
        total_range_count = sum(overall_range_data_dict.values())

        for key in overall_range_data_dict.keys():
            range_data_dict = {"range": key, "count": overall_range_data_dict[key]}
            percent = 0.0

            if int(total_range_count) != 0:
                percent = (float(range_data_dict["count"]) / total_range_count) * 100

            range_data_dict["range_percentage"] = percent
            overall_range_data.append(range_data_dict)

        combined_overall_data["overall_unclubbed_range_data"] = overall_range_data
        if overall_range_data:
            overall_clubbed_range_data = club_range(overall_range_data)
            combined_overall_data["overall_range_data"] = overall_clubbed_range_data
        else:
            combined_overall_data["overall_range_data"] = []

        overall_length_sizes_list.sort()
        overall_size_uniformity_status, combined_overall_data["overall_size_uniformity_data"] = size_uniformity_model(
            overall_length_sizes_list)

        status, grade_range_data, bold_count, others_count = get_grade_details(overall_width_sizes_list, commodity_id)
        if status == 3:
            combined_overall_data["overall_range_grade_details"] = grade_range_data
            combined_overall_data["bold_ratio"] = "Bold : others  -  " + str(bold_count) + " : " + str(others_count)

        combined_overall_data["overall_total_count"] = len(overall_length_sizes_list)

        combined_overall_data["overall_max_length"], combined_overall_data["overall_min_length"], combined_overall_data[
            "overall_avg_lenght"] = find_length_min_max_avg(overall_size_data)
        combined_overall_data["overall_max_width"], combined_overall_data["overall_min_width"], combined_overall_data[
            "overall_avg_width"] = find_width_min_max_avg(overall_size_data)

        combined_overall_data = json.dumps(combined_overall_data, indent=1)
        combined_overall_data = json.loads(combined_overall_data)
        combined_overall_data = str(combined_overall_data).replace("'", '"')

        update_quick_imgpro_data(ticket_id, ticketid_data, combined_overall_data)
        logging.info("DB connection closed...!!!")
        mydb.close()
        logging.info(f"Time taken for Quick Analysis - {t.time() - start_time}")
        return {"message": "Quick image process done for the ticket-id : " + str(ticket_id), "status": "1"}


@app.route('/quick_imgpro/v1.0_touched_image_process', methods=['GET', 'POST'])
def touched_size_range():
    """
    API to detect the objects and calculate the size of objects.
    :return: Return the dict of final output data.
    """

    if request.method == 'POST':
        start_time = t.time()
        ticket_id = request.form['ticket_id']
        user_id = request.form['user_id']
        server = request.form['server']
        if server != 'uat':
            return {"message": "Hitting UAT server, but request server key-value is not 'uat'", "status": "0"}

        # global mydb
        mydb, connected = db_connect(server)
        rows = get_image_details_quick_imgpro(ticket_id)
        print("Images  : ", rows)
        if not rows:
            logging.info(f'No images present for the ticket id: {ticket_id}')
            return {"message": "No Images for the ticket-id : " + str(ticket_id), "status": "0"}

        coin_diameter = get_coin_diameter_quick_imgpro(ticket_id)

        coin_diameter = coin_diameter[0][0]
        if coin_diameter:
            try:
                coin_diameter = float(coin_diameter)
            except:
                coin_diameter = 26.8
        else:
            coin_diameter = 26.8

        commodity_id = get_commodity_id(ticket_id)[0]
        commodity_id = str(commodity_id[0])

        ticketid_data = {}
        overall_length_sizes_list = []
        overall_width_sizes_list = []
        overall_size_data = []
        overall_range_data = []
        overall_range_data_dict = {}
        image_name_list = []
        for row in rows:
            image_id = str(row[0])
            url_image_path = str(row[1])
            image_weight = float(row[2])
            img_data = {}
            try:
                # Download pending image and saving in input_images folder
                downloaded_image_path = download_image(ticket_id, image_id, url_image_path, server)

            except Exception as e:
                logging.error(e)
                logging.error('Error in downloading the image.. \nThis is Production server..'
                              ' may be live code will execute')
                continue

            # Size calculation part
            img_data["size_status"], img_data["size_data"], length_sizes_list, width_sizes_list = touched_size_model(
                ticket_id, image_id, downloaded_image_path, commodity_id, coin_diameter)
            img_data["size_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            length_sizes_list.sort()
            width_sizes_list.sort()

            if img_data["size_status"] == 3:
                overall_length_sizes_list = overall_length_sizes_list + length_sizes_list
                overall_size_data = overall_size_data + img_data["size_data"]

                overall_width_sizes_list = overall_width_sizes_list + width_sizes_list

                img_data["size_uniformity_status"], img_data["size_uniformity_data"] = size_uniformity_model(
                    length_sizes_list)
                img_data["size_uniformity_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Range calculation part..
                img_data["range_status"], img_data["range_data"] = range_model(length_sizes_list)
                img_data["range_res_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                img_data["image_total_weight"] = image_weight
                img_data["image_total_count"] = len(length_sizes_list)

                status, grade_range_data, bold_count, others_count = get_grade_details(width_sizes_list, commodity_id)
                if status == 3:
                    img_data["image_range_grade_details"] = grade_range_data
                    img_data["bold_ratio"] = "Bold : others  -  " + str(bold_count) + " : " + str(others_count)

                img_data["image_max_length"], img_data["image_min_length"], img_data[
                    "image_avg_lenght"] = find_length_min_max_avg(img_data["size_data"])
                img_data["image_max_width"], img_data["image_min_width"], img_data[
                    "image_avg_width"] = find_width_min_max_avg(img_data["size_data"])

                img_data["image_unclubbed_range_data"] = img_data["range_data"]
                if img_data["range_data"]:
                    img_data["range_data"] = club_range(img_data["range_data"])
                else:
                    img_data["range_data"] = []

                ticketid_data[image_id] = img_data

                if img_data["range_status"] == 3:
                    for r in img_data["range_data"]:
                        range_ = r["range"]
                        count = r["count"]
                        overall_range_data_dict[range_] = overall_range_data_dict.get(range_, 0) + count

                image_name_list.append(str(ticket_id) + '_' + str(image_id) + '.png')

            else:
                logging.error(f"Image Error for the Ticket_Image_id : {ticket_id}_{image_id}")

            os.remove(downloaded_image_path)

        upload_output_image(ticket_id, image_name_list)

        ticketid_data = json.dumps(ticketid_data, indent=1)
        ticketid_data = json.loads(ticketid_data)
        ticketid_data = str(ticketid_data).replace("'", '"')

        logging.info("Combined overall data started...")
        combined_overall_data = {}
        combined_overall_data["overall_size_data"] = overall_size_data

        total_range_count = sum(overall_range_data_dict.values())
        for key in overall_range_data_dict.keys():
            range_data_dict = {}
            range_data_dict["range"] = key
            range_data_dict["count"] = overall_range_data_dict[key]
            percent = 0.0
            if int(total_range_count) != 0:
                percent = (float(range_data_dict["count"]) / total_range_count) * 100
            range_data_dict["range_percentage"] = percent
            overall_range_data.append(range_data_dict)

        combined_overall_data["overall_unclubbed_range_data"] = overall_range_data
        if overall_range_data != []:
            overall_clubbed_range_data = club_range(overall_range_data)
            combined_overall_data["overall_range_data"] = overall_clubbed_range_data
        else:
            combined_overall_data["overall_range_data"] = []

        overall_length_sizes_list.sort()
        overall_size_uniformity_status, combined_overall_data["overall_size_uniformity_data"] = size_uniformity_model(
            overall_length_sizes_list)

        status, grade_range_data, bold_count, others_count = get_grade_details(overall_width_sizes_list, commodity_id)
        if status == 3:
            combined_overall_data["overall_range_grade_details"] = grade_range_data
            combined_overall_data["bold_ratio"] = "Bold : others  -  " + str(bold_count) + " : " + str(others_count)

        combined_overall_data["overall_total_count"] = len(overall_length_sizes_list)

        combined_overall_data["overall_max_length"], combined_overall_data["overall_min_length"], combined_overall_data[
            "overall_avg_lenght"] = find_length_min_max_avg(overall_size_data)
        combined_overall_data["overall_max_width"], combined_overall_data["overall_min_width"], combined_overall_data[
            "overall_avg_width"] = find_width_min_max_avg(overall_size_data)

        combined_overall_data = json.dumps(combined_overall_data, indent=1)
        combined_overall_data = json.loads(combined_overall_data)
        combined_overall_data = str(combined_overall_data).replace("'", '"')

        update_quick_imgpro_data(ticket_id, ticketid_data, combined_overall_data)
        logging.info("DB connection closed...!!!")
        mydb.close()
        logging.info(f"Time taken for Quick Analysis {t.time() - start_time}")
        return {"message": "Quick image process done for the ticket-id : " + str(ticket_id), "status": "1"}

@app.route('/quick_imgpro/test_quick_imgpro_app', methods=['GET', 'POST'])
def test():
    return 'The quick_imgpro_app is up and running. Send a POST request'
    

# anomaly_model = pickle.loads(open(anomaly_model_path, "rb").read())

if __name__ == '__main__':
    # Trying to connect Database

    configured = check_configurations()
    app.run(port_num)
