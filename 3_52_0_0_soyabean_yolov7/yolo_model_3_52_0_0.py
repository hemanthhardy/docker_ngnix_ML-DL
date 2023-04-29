# Importing required libraries
import json
import time
import os
import logging
import requests

import cv2
import image_slicer
import torch
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import urllib.request
from datetime import datetime
from flask import Flask, request
from numpy import random
from shutil import make_archive

from models.experimental import attempt_load
from config import *
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')

app = Flask(__name__)


# gpus = tf.config.list_physical_devices('GPU')
#
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1500)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


def coin_detection(image_path, ticket_id, image_id, coin_diameter):
    # Detect the coin from image and find the pixel per metrix for the reference
    coin_start = time.time()
    logging.info(f"Coin detection started...")

    if coin_diameter is None:
        coin_diameter = 26.8
    ref_width = float(coin_diameter)  # Reference dia of the coin

    logging.info(f'1 - {time.time()}')
    image = cv2.imread(image_path)
    logging.info(f'2 - {time.time()}')
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    logging.info(f'3 - {time.time()}')
    blurred = cv2.medianBlur(gray, 25)
    logging.info(f'4 - {time.time()}')
    circles = cv2.HoughCircles(blurred,
                               cv2.HOUGH_GRADIENT,
                               1,
                               minDist=100,
                               param1=30,
                               param2=50,
                               minRadius=100,
                               maxRadius=300)
    logging.info(f'5 - {time.time()}')

    try:
        if circles is None:
            logging.error(f"No coin detected...")
            coin_end = time.time()
            total_time = coin_end - coin_start
            logging.info(f'Total time taken in coin detection {total_time:.2f}')
            return default_pixel_per_metrix
    except Exception as e:
        pass

    if circles is not None:
        # Using the circle dia calculate the pixel per metrics
        # Draw the circle around the coin and overwrite input image
        logging.info(f'Draw the circle on the coin...')
        logging.info(f'6 - {time.time()}')
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]
        logging.info(f'7 - {time.time()}')
        cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        logging.info(f'8 - {time.time()}')
        pixel_dia = circles[0][0][-1] * 2
        pixel_per_metrix = pixel_dia / ref_width
        logging.info(f'pixel_per_metrix - {pixel_per_metrix}')
        logging.info(f'9 - {time.time()}')

        output_image_path = slice_image_dir + str(ticket_id) + '/'
        logging.info(f'10 - {time.time()}')

        # Create a separate directory for this ticket to store crop images
        if not os.path.exists(output_image_path):
            os.mkdir(output_image_path)

        output_image_path = output_image_path + str(image_id) + '/'
        if not os.path.exists(output_image_path):
            os.mkdir(output_image_path)

        logging.info(f'11 - {time.time()}')
        cv2.imwrite(output_image_path + 'coin_detect.jpg', image)

        coin_end = time.time()
        total_time = coin_end - coin_start
        logging.info(f'Total time taken in coin detection {total_time:.2f}')
        return pixel_per_metrix
    coin_end = time.time()
    total_time = coin_end - coin_start
    logging.info(f'Total time taken in coin detection {total_time:.2f}')
    return default_pixel_per_metrix


def crop_test_image(test_img_path, image_id, ticket_id, coin_diameter):
    try:
        crop_start = time.time()
        # Detect the coin and crop the image into 4 slices
        logging.info(f"Slicing the image into 4 pieces to predict the grains...")

        try:
            pixel_per_metrix = coin_detection(test_img_path, ticket_id, image_id, coin_diameter)
        except:
            pixel_per_metrix = default_pixel_per_metrix

        slice_image_path = slice_image_dir + str(ticket_id) + '/' + str(image_id) + '/'

        if not os.path.exists(slice_image_path):
            # Create a directory if it's not exists
            os.mkdir(slice_image_path)

        image_to_slice = test_img_path
        if os.path.exists(slice_image_path + 'coin_detect.jpg'):
            # Take the image with coin detection if its detected
            image_to_slice = slice_image_path + 'coin_detect.jpg'

        tiles = image_slicer.slice(image_to_slice, 4, save=False)
        for image_type in image_types_list:
            if not os.path.exists(slice_image_path + image_type):
                # Create a directory if it's not exists
                os.mkdir(slice_image_path + image_type)

            image_slicer.save_tiles(tiles, directory=slice_image_path + image_type, prefix='slice', format='jpeg')

        if os.path.exists(slice_image_path + 'coin_detect.jpg'):
            # Remove the coin detection image from the output folder
            os.remove(slice_image_path + 'coin_detect.jpg')

        crop_end = time.time()
        total_crop = crop_end - crop_start
        logging.info(f'Total cropping and deleting time {total_crop:.2f}')

        return True, slice_image_path, pixel_per_metrix
    except Exception as e:
        logging.error(f'{e}')
        crop_end = time.time()
        total_crop = crop_end - crop_start
        logging.info(f'Total cropping and deleting time {total_crop:.2f}')
        return False, '', default_pixel_per_metrix


def run_yolov7_model(slice_image_path, pixel_per_metrix):
    try:
        logging.info('Started analysing the cropped image.......')
        conf_thres = 0.25
        iou_thres = 0.45

        half = device.type != 'cpu'  # half precision only supported on CUDA

        logging.info("Model traced.....")

        if half:
            model.half()  # to FP16
        logging.info(f"Model traced.....{slice_image_path} - {imgsz}  - {stride}")
        # Set Dataloader
        dataset = LoadImages(slice_image_path + 'original/', img_size=imgsz, stride=stride)
        logging.info("Model traced 1.....")
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        logging.info("Model traced 2.....")
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        old_img_w = old_img_h = imgsz
        old_img_b = 1
        logging.info(f"Detected the dataset for sliced images")
        final_result = {}
        length_width_list = []
        length_list = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                print("pred", pred)
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}, "  # add to string

                    good_image = cv2.imread(slice_image_path + 'good/' + p.name)
                    semi_image = cv2.imread(slice_image_path + 'semi_processable/' + p.name)
                    unused_image = cv2.imread(slice_image_path + 'unused/' + p.name)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print('detected')
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        coordinate = []
                        for tensor_coord in xyxy:
                            coordinate.append(int((str(tensor_coord).split('(')[1]).split('.')[0]))

                        pred_class = names[int(cls)]
                        length_all = (coordinate[2] - coordinate[0]) / pixel_per_metrix
                        width_all = (coordinate[3] - coordinate[1]) / pixel_per_metrix

                        print('label', f'{names[int(cls)]} {conf:.2f} {length_all:.2f} {width_all:.2f}')
                        label = f'{names[int(cls)]} {length_all:.2f} {width_all:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        if pred_class.upper() not in ['STONES', 'STICKS', 'COIN'] and pred_class.find('admixture') < 0:
                            length = (coordinate[2] - coordinate[0]) / pixel_per_metrix
                            width = (coordinate[3] - coordinate[1]) / pixel_per_metrix
                            length_width_list.append({"length": length, "width": width})
                            length_list.append(length)
                            if pred_class.upper() == 'GOOD':
                                plot_one_box(xyxy, good_image, label=label, color=colors[int(cls)], line_thickness=1)
                            else:
                                plot_one_box(xyxy, semi_image, label=label, color=colors[int(cls)], line_thickness=1)
                                pass
                        else:
                            plot_one_box(xyxy, unused_image, label=label, color=colors[int(cls)], line_thickness=1)
                            pass

                    cv2.imwrite(slice_image_path + 'good/' + p.name, good_image)
                    cv2.imwrite(slice_image_path + 'semi_processable/' + p.name, semi_image)
                    cv2.imwrite(slice_image_path + 'unused/' + p.name, unused_image)
                    cv2.imwrite(slice_image_path + 'original/' + p.name, im0)

                    output_list = s.split(',')

                    for value in output_list:
                        if value == ' ':
                            continue
                        value = value.strip()
                        dat_list = value.split(' ')
                        if dat_list[1] in list(final_result.keys()):
                            final_result[dat_list[1]] += int(dat_list[0])
                        else:
                            final_result[dat_list[1]] = int(dat_list[0])

        # mapping class_id to quality names
        ids_labels = open(class_path, 'r').read()
        classes = {}
        labels = ids_labels.split('\n')
        for i in labels:
            if i != '':
                classes[int(i.split(',')[0])] = i.split(',')[1]

        logging.info(f'classes: {classes}')

        class_map_data = {}
        for key, value in classes.items():
            if value not in final_result.keys():
                final_result[value] = 0
            class_map_data[key] = final_result[value]

        logging.info(f'class_map_data: {class_map_data}')
        length_width_data = [length_width_list, length_list]
        return class_map_data, length_width_data
    except Exception as e:
        logging.error(f"Model not loading: {e}")
        return {}, [[()], []]


def cal_approx_wt(total_wt, loc_to_db_class_map, loc_class_counts, db_class_counts, class_id_to_weight_class_map,
                  class_id_to_factor_value_map):
    logging.info(f"Weight calculation started....!!!! {class_id_to_weight_class_map}")
    ns_weight = 0.0
    sum_of_counts_x_factor = 0.0
    loc_class_wt_ns = {}
    loc_class_fv_s = {}

    for key in class_id_to_weight_class_map.keys():
        if class_id_to_weight_class_map[key] != 1 and class_id_to_weight_class_map[key] != 2:
            temp_wt = (loc_class_counts[key] * class_id_to_factor_value_map[key])
            loc_class_wt_ns[key] = temp_wt
            ns_weight = ns_weight + temp_wt
            logging.info(f"ns_weight : {ns_weight}")
        else:
            temp_fv = loc_class_counts[key] * class_id_to_factor_value_map[key]
            loc_class_fv_s[key] = temp_fv
            sum_of_counts_x_factor = sum_of_counts_x_factor + (temp_fv)
            logging.info("sum_of_counts_x_factor : {sum_of_counts_x_factor}")

    logging.info(f"(Factor_value * count) - value :  {loc_class_fv_s}")

    sample_wt = abs(total_wt - ns_weight)
    if sample_wt < 0.0:
        logging.error(f"Sample_wt is less_than zero..  Means non_sample_weight > total weight")
        return {}

    logging.info(f"\n\nTotal_weight: {total_wt} non_sample_weight: {ns_weight} sample_weight: {sample_wt}\n\n")
    if sum_of_counts_x_factor == 0.0:
        x = 0.0
    else:
        x = sample_wt / sum_of_counts_x_factor

    loc_class_wt_s = {}
    for key in loc_class_fv_s.keys():
        if class_id_to_weight_class_map[key] == 1 or class_id_to_weight_class_map[key] == 2:
            loc_class_wt_s[key] = x * loc_class_fv_s[key]

    logging.info(f"non_sample_weights: {loc_class_wt_ns} \n Sample_weights: {loc_class_wt_s}")

    loc_class_wt = loc_class_wt_ns.copy()
    loc_class_wt.update(loc_class_wt_s)

    logging.info(f"Local id mapped wt: {loc_class_wt}")

    db_class_wt = {}
    for key in loc_to_db_class_map.keys():
        db_id = int(loc_to_db_class_map[key])
        if db_id in db_class_wt.keys():
            db_class_wt[db_id] = db_class_wt[db_id] + loc_class_wt[key]
        else:
            db_class_wt[db_id] = loc_class_wt[key]

    logging.info(f"DB id mapped wt: {db_class_wt}")
    return db_class_wt


def merge_cropped_image(slice_image_path, ticket_id, image_id):
    try:
        # Merge the sliced images with the prediction
        logging.info(f"Merging the output cropped images")
        for image_type in image_types_list:
            image_list = os.listdir(slice_image_path + image_type + '/')
            image_1_size = Image.open(slice_image_path + image_type + '/' + image_list[0]).size

            new_image = Image.new('RGB', (2 * image_1_size[0], 2 * image_1_size[1]), (250, 250, 250))

            new_image.paste(Image.open(slice_image_path + image_type + '/' + 'slice_01_01.jpg'), (0, 0))
            new_image.paste(Image.open(slice_image_path + image_type + '/' + 'slice_01_02.jpg'), (image_1_size[0], 0))
            new_image.paste(Image.open(slice_image_path + image_type + '/' + 'slice_02_01.jpg'), (0, image_1_size[1]))
            new_image.paste(Image.open(slice_image_path + image_type + '/' + 'slice_02_02.jpg'), (image_1_size[0],
                                                                                                  image_1_size[1]))

            if not os.path.exists(output_merged_image_path + ticket_id + '/'):
                # Create a directory if it's not exists
                os.mkdir(output_merged_image_path + ticket_id + '/')
            new_image.save(output_merged_image_path + str(ticket_id) + '/' +
                           str(ticket_id) + '_' + str(image_id) + '_' + str(image_types[image_type]) + ".jpg", "JPEG")
        shutil.rmtree(slice_image_path)

    except Exception as e:
        logging.error(f'{e}')


def control_function(t_id, image_id, total_wt, cat_id, comm_id, var_id, svar_id, test_img_path, coin_diameter):
    logging.info(
        f'Processing... ticket_id: {t_id} - category_id: {cat_id} - commodity_id: {comm_id}'
        f' - variety_id: {var_id} - sub-variety_id {svar_id}')

    status, slice_image_path, pixel_per_metrix = crop_test_image(test_img_path, image_id, t_id, coin_diameter)
    class_counts, length_width_list = run_yolov7_model(slice_image_path, pixel_per_metrix)
    merge_cropped_image(slice_image_path, t_id, image_id)

    logging.info('Text file details FYR: ')
    ids_labels = open(class_path, 'r').read()
    classes = {}
    loc_to_db_class_map = {}
    class_id_to_weight_class_map = {}
    class_id_to_factor_value_map = {}
    labels = ids_labels.split('\n')

    for row in labels:
        if row != '':
            classes[int(row.split(',')[0])] = row.split(',')[1]
            loc_to_db_class_map[int(row.split(',')[0])] = row.split(',')[2]
            class_id_to_weight_class_map[int(row.split(',')[0])] = int(row.split(',')[3])
            class_id_to_factor_value_map[int(row.split(',')[0])] = float(row.split(',')[4])

    logging.info(f'classes: {classes}')
    logging.info(f'loc_to_db_class_map: {loc_to_db_class_map}')
    logging.info(f"quality_id - weight_id: {class_id_to_weight_class_map}")
    logging.info(f"quality_id - factor value: {class_id_to_factor_value_map}")

    # Counting the only predicted detections
    counts = class_counts
    logging.info("Counts : ", counts)

    # Changing local class ids with counts to DB class ids with counts 
    db_class_counts = {}
    for key in loc_to_db_class_map.keys():
        db_id = int(loc_to_db_class_map[key])
        if db_id in db_class_counts.keys():
            db_class_counts[db_id] = db_class_counts[db_id] + counts[key]
        else:
            db_class_counts[db_id] = counts[key]

    db_class_wt = cal_approx_wt(total_wt, loc_to_db_class_map, counts, db_class_counts, class_id_to_weight_class_map,
                                class_id_to_factor_value_map)

    logging.info(f'Dict with db classes map: {db_class_counts}')
    return db_class_counts, db_class_wt, length_width_list


def quality_function(t_id, image_id, cat_id, comm_id, var_id, svar_id, test_img_path, total_wt, coin_diameter):
    total_wt = float(total_wt)

    final_data = []

    db_class_counts, db_class_wt, length_width_list = control_function(t_id, image_id, total_wt, cat_id, comm_id,
                                                                       var_id, svar_id, test_img_path, coin_diameter)

    total_count = sum(db_class_counts.values())
    db_classname_map_path = inference_graph_path + db_classname_map_file
    db_classname_map = {}
    db_classname = open(db_classname_map_path, 'r').read()

    for line in db_classname.split('\n'):
        if line != '':
            key, value = line.split(',')
            db_classname_map[int(key)] = str(value)

    logging.info(f"ID - class_name: {db_classname_map}")
    logging.info(f'total_count: {total_count}')

    for key in db_class_counts.keys():

        if total_count == 0:
            count_percent = 0.0
        else:
            count_percent = float((db_class_counts[key] / total_count) * 100)

        if db_class_wt == {}:
            wt_percent = 0.0
            single_quality_value = {"quality_parameter_id": key,
                                    "quality_parameter_name": db_classname_map[key],
                                    "count": db_class_counts[key],
                                    "count_percentage": count_percent,
                                    "weight": 0.0,
                                    "weight_percentage": wt_percent}
        else:
            wt_percent = float((db_class_wt[key] / total_wt) * 100)
            single_quality_value = {"quality_parameter_id": key,
                                    "quality_parameter_name": db_classname_map[key],
                                    "count": db_class_counts[key],
                                    "count_percentage": count_percent,
                                    "weight": db_class_wt[key],
                                    "weight_percentage": wt_percent}

        final_data.append(single_quality_value)

    final_data = json.dumps(final_data, indent=1)
    final_data = json.loads(final_data)
    final_data = str(final_data).replace("'", '"')
    return 3, final_data, db_class_counts, db_class_wt, db_classname_map, length_width_list


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
    path_to_download = input_dir_path + str(ticket_id) + '_' + str(image_id) + ".jpg"
    logging.info(f"path_to_download : {path_to_download}")
    urllib.request.urlretrieve(url_to_download, path_to_download)
    logging.info(f'Image downloaded!!!')
    return path_to_download


def upload_output_image(ticket_id, server):
    """
    Function to upload the output image to image server.
    @param server: Server name either UAT or LIVE.
    @param ticket_id: Ticket ID of the request.
    """
    try:

        make_archive(output_merged_image_path + str(ticket_id),
                     'zip',
                     root_dir=output_merged_image_path + str(ticket_id))

        upload_image_url = image_server[server] + 'output_images/upload_output_images/'

        files = {"output_images": open(output_merged_image_path + str(ticket_id) + '.zip', 'rb')}
        data = {"request_id": ticket_id}
        getdata = requests.post(upload_image_url, data=data, files=files)

        data = json.loads(getdata.text)

        if data['status']:
            shutil.rmtree(output_merged_image_path + str(ticket_id))
            os.remove(output_merged_image_path + str(ticket_id) + '.zip')

    except Exception as e:
        logging.error(f'{e}')


@app.route('/3_52_0_0/v1.0_predict_yolo_3_52_0_0', methods=['GET', 'POST'])
def quality_prediction():
    if request.method == 'POST':
        start = time.time()
        ticket_id = request.form['ticket_id']
        category_id = request.form['category_id']
        commodity_id = request.form['commodity_id']
        variety_id = request.form['variety_id']
        sub_variety_id = request.form['sub_variety_id']
        image_details = json.loads(request.form['image_details'])
        image_wise_data = json.loads(request.form['image_wise_data'])
        coin_diameter = request.form['coin_diameter']
        server = request.form['server']

        combined_actual_total_weight = 0.0
        overall_length_sizes_list = []
        overall_size_data = []
        overall_quality_count = {}
        overall_quality_wt = {}
        invalid_wt_check = 0
        db_class_counts = {}
        db_classname_map = {}

        logging.info(f'Image Details - {image_details}')

        for image_detail in image_details:
            print(image_detail)
            image_id = str(image_detail[0])
            url_image_path = str(image_detail[1])
            image_wise_total_weight = str(image_detail[2])
            combined_actual_total_weight = combined_actual_total_weight + float(image_wise_total_weight)

            try:
                logging.info(f'save the image in path')
                image_path = download_image(ticket_id, image_id, url_image_path, server)
                logging.info(f'Image saved...')
            except Exception as e:
                logging.error(f'{e}')
                logging.error(f'Unable to download input image')
                image_path = ''
                pass

            quality_status, quality_data, db_class_counts, db_class_wt, \
                db_classname_map, length_width_data = quality_function(ticket_id,
                                                                       image_id,
                                                                       category_id,
                                                                       commodity_id,
                                                                       variety_id,
                                                                       sub_variety_id,
                                                                       image_path,
                                                                       image_wise_total_weight,
                                                                       coin_diameter)

            quality_res_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Quality Data : {quality_data}")

            if image_wise_data is None or image_wise_data == {} or image_wise_data == 0:
                image_wise_data = {}
                image_id_data = {"quality_data": json.loads(quality_data),
                                 "quality_status": quality_status,
                                 "quality_res_time": quality_res_time,
                                 "actual_image_weight": image_wise_total_weight}
                image_wise_data[image_id] = image_id_data
            else:
                if str(image_id) in image_wise_data.keys():
                    image_wise_data[image_id]["quality_data"] = json.loads(quality_data)
                    image_wise_data[image_id]["quality_status"] = quality_status
                    image_wise_data[image_id]["quality_res_time"] = quality_res_time
                    image_wise_data[image_id]["actual_image_weight"] = image_wise_total_weight
                else:
                    image_id_data = {"quality_data": json.loads(quality_data),
                                     "quality_status": quality_status,
                                     "quality_res_time": quality_res_time,
                                     "actual_image_weight": image_wise_total_weight}

                    image_wise_data[image_id] = image_id_data

            overall_length_sizes_list.extend(length_width_data[1])
            overall_size_data.extend(length_width_data[0])

            if quality_status == 3:
                for k in db_class_counts.keys():
                    overall_quality_count[k] = overall_quality_count.get(k, 0) + db_class_counts[k]
                    if db_class_wt == {}:
                        invalid_wt_check = 1
                        overall_quality_wt[k] = overall_quality_wt.get(k, 0) + 0.0
                    else:
                        overall_quality_wt[k] = overall_quality_wt.get(k, 0) + db_class_wt[k]
            else:
                logging.error(f"Quality status not 3")

        final_data = {'status': 3,
                      'overall_quality_count': overall_quality_count,
                      'overall_quality_wt': overall_quality_wt,
                      'combined_actual_total_weight': combined_actual_total_weight,
                      'overall_length_sizes_list': overall_length_sizes_list,
                      'overall_size_data': overall_size_data,
                      'invalid_wt_check': invalid_wt_check,
                      'db_class_counts': db_class_counts,
                      'db_classname_map': db_classname_map
                      }

        try:
            upload_output_image(ticket_id, server)
        except Exception as e:
            logging.error(f'{e}')

        logging.info(f'final_data: {final_data}')
        logging.info('Soyabean model completed.....')

        end = time.time()
        total_time = end - start
        logging.info(f"Total Time taken by Soyabean model: {total_time:.2f} sec")

        return final_data

    return {"status": "Request method should be POST not GET"}


device = ''
device = select_device(device)
model = attempt_load(model_file_name, map_location=device)
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
model = TracedModel(model, device, imgsz)

if __name__ == '__main__':
    app.run(port=port_num)
