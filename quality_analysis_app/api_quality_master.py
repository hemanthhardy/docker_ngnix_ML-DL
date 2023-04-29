# Import required packages
import json
import logging
import os
import time as t

from datetime import datetime
from flask import Flask, request
from config import *
from utils.quality_queries import *
from utils.range_model import *
from utils.size_uniformity_model import *


logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')

app = Flask(__name__)


def range_model(length_sizes_list):
    """
    Function to calculate the length range of all the grains.
    :param length_sizes_list: List of grains length to calculate the range.
    :return: Return the list of dictionaries with range details.
    """

    try:
        data = range_calculation(length_sizes_list)
        return 3, data

    except Exception as e:
        logging.error(f'Error in range model - {e}')
        return 4, []


def size_uniformity_model(sizes_list):
    """
    Function to uniformize the detected object size and calculate mean, variance,
    standard_deviation, window size, range intervel and ranges.
    :param sizes_list: List of detected object sizes.
    :return: Return the Dictionary of values.
    """

    try:
        data = size_uniformity_master_function(sizes_list)
        return 3, data
    except Exception as e:
        logging.error(f'Error in size uniformity model - {e}')
        return 4, {}


def club_range(range_data):
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
            return [{'range': '0.0 - 60.0',
                     'count': 0,
                     'range_percentage': 0.0}]

    if exists:
        range_ = range_ + range_data[0]['range'].split("-")[0][:-1]
        dic = {'range': range_,
               'count': count_,
               'range_percentage': clubbed_percent}

        range_data.insert(0, dic)

    checker = 0
    clubbed_percent = 0.0
    range_ = ' - 60.0'
    count_ = 0
    exists = False
    while range_data[-1]['range_percentage'] < range_clubbing_percent:
        exists = True
        if range_data[-1]['range_percentage'] == 0.0 and checker == 0:
            del range_data[-1]
            range_ = ' - ' + range_data[-1]['range'].split("-")[-1][1:]

        else:
            checker = 1
            clubbed_percent += range_data[-1]['range_percentage']
            count_ += range_data[-1]['count']
            del range_data[-1]

    if exists:
        range_ = range_data[-1]['range'].split("-")[-1][1:] + range_
        dic = {'range': range_,
               'count': count_,
               'range_percentage': clubbed_percent}
        range_data.append(dic)
    logging.info(f'Clubbing Completed....{range_data}')
    return range_data


def find_width_min_max_avg(overall_size_data):
    """
    Function to calculate the average, minimum and maximum values of width.
    :param overall_size_data: Overall detected objects size data.
    :return: Return the minimum, maximum and average values of width.
    """

    if not overall_size_data:
        return 0.0, 0.0, 0.0

    mini = 10000.0
    maxi = 0.0
    sum = 0.0
    for i in overall_size_data:
        wid = float(i['width'])
        sum = sum + wid
        if wid > maxi:
            maxi = wid
        if wid < mini:
            mini = wid

    aver = sum / len(overall_size_data)
    logging.info(f"WIDTH :   min-", {mini}, '     maxi-', {maxi}, '    average-', {aver})
    return maxi, mini, aver


def find_length_min_max_avg(overall_size_data):
    """
    Function to calculate the average, minimum and maximum values of width.
    :param overall_size_data: Overall detected objects size data.
    :return: Return the minimum, maximum and average values of width.
    """

    if not overall_size_data:
        return 0.0, 0.0, 0.0

    mini = 10000.0
    maxi = 0.0
    sum = 0.0
    for i in overall_size_data:
        leng = float(i['length'])
        sum = sum + leng
        if leng > maxi:
            maxi = leng
        if leng < mini:
            mini = leng

    aver = sum / len(overall_size_data)
    logging.info("LENGTH :  min-", {mini}, '     maxi-', {maxi}, '    average-', {aver})
    return maxi, mini, aver


def quality_yolo_models(ticket_id, category_id, commodity_id, variety_id, sub_variety_id,
                        image_details, image_wise_data, coin_diameter, server):
    """
    Function to call the YOLO model to detect the object and calculate the size of objects.
    :param image_wise_data:
    :param ticket_id: Ticket ID of this request.
    :param category_id: Category ID of this request.
    :param commodity_id: Commodity ID of this request.
    :param variety_id: Variety ID of this request.
    :param sub_variety_id: Sub Variety ID of this request.
    :param image_details: Complete image details from database.
    :param coin_diameter: Coin Diameter entered by the user.
    :param server: Name of the server handling the request.
    :return: Return the prediction and size data of the request.
    """

    try:
        logging.info(f"Quality yolo model started..")
        commodity_details = category_id + '_' + commodity_id + '_' + variety_id + '_' + sub_variety_id

        logging.info(f'Commodity_details - {commodity_details}')
        if server == 'uat':
            port_no = yolo_port_no_map_uat[commodity_details][0]
            url = yolo_port_no_map_uat[commodity_details][1]
        elif server == 'live':
            port_no = yolo_port_no_map_live[commodity_details][0]
            url = yolo_port_no_map_live[commodity_details][1]
        else:
            print("server not correct")
        logging.info(f'{url}{port_no}')

        
        model_url = url + port_no + '/' + commodity_details + '/v1.0_predict_yolo' + '_' + commodity_details
        logging.info(f'{model_url}')

        form_data = {"ticket_id": ticket_id,
                     "category_id": category_id,
                     "commodity_id": commodity_id,
                     "variety_id": variety_id,
                     "sub_variety_id": sub_variety_id,
                     "image_details": json.dumps(image_details) if image_details else json.dumps({}),
                     'image_wise_data': json.dumps(image_wise_data) if image_wise_data else json.dumps({}),
                     "coin_diameter": coin_diameter,
                     "server": server}

        logging.info(f'Image Details - {image_details}')
        logging.info(f'Form Data - {form_data}')

        get_data = requests.post(model_url, form_data)
        data = json.loads(get_data.text)

        logging.info(f'{data}')
        if data["status"] == 3:
            return data["status"], data["overall_quality_count"], data["overall_quality_wt"],\
                data["combined_actual_total_weight"], data["overall_length_sizes_list"], data["overall_size_data"],\
                data['invalid_wt_check'], data['db_class_counts'], data['db_classname_map']
        else:
            return 4, {}, {}, {}, {}, []

    except Exception as e:
        logging.error(f'Error in Yolo model - {e}')
        return 4, {}, {}, {}, {}, []


# Flask API for Defect Analysis
@app.route('/quality/v1.0_quality_yolo_models', methods=['GET', 'POST'])
def quality_yolo():
    """
    Flask API to get ticket_id, user_id, server name from request.
    Load the yolo model based on commodity and process the predicted model results.
    Form the result data in a json format and update the date-base with result.
    :return: Return the message with status code.
    """

    start = t.time()
    if request.method == 'POST':
        # Get the request details
        ticket_id = request.form['ticket_id']
        user_id = request.form['user_id']
        server = request.form['server']

        mydb, connected = db_connect_quality(server)

        # Get the details of the request ticket using ticket_id
        ticket_details = get_ticketid_details_quality(ticket_id)[0]
        if not ticket_details:
            logging.error(f'Requested ticket id is not created in DB !!!')
            return {"message": "ticket-id : " + str(ticket_id) + " not exists", "status": "0"}

        logging.info(f'Ticket Details - {ticket_details}')

        ticket_id = str(ticket_details[0])
        category_id = str(ticket_details[1])
        commodity_id = str(ticket_details[2])
        variety_id = str(ticket_details[3])
        sub_variety_id = str(ticket_details[4])
        coin_diameter = str(ticket_details[7])

        logging.info(f'coin_diameter - {coin_diameter}')

        if ticket_details[5] is None:
            image_wise_data = {}
            combined_data = {}
        else:
            image_wise_data = json.loads(str(ticket_details[5]))
            combined_data = json.loads(str(ticket_details[6]))

        # Get the image details from database
        image_details = get_image_details_quality(ticket_id)
        if not image_details:
            logging.error(f'No images present for the ticket id: {ticket_id}')
            return {"message": "No Images details for the ticket-id : " + str(ticket_id), "status": "0"}

        logging.info(f'Image Details - {image_details}')

        quality_status, overall_quality_count, overall_quality_wt,\
            combined_actual_total_weight, overall_length_sizes_list,\
            overall_size_data, invalid_wt_check, db_class_counts, db_classname_map = quality_yolo_models(ticket_id,
                                                                                                         category_id,
                                                                                                         commodity_id,
                                                                                                         variety_id,
                                                                                                         sub_variety_id,
                                                                                                         image_details,
                                                                                                         image_wise_data,
                                                                                                         coin_diameter,
                                                                                                         server)

        overall_quality_data = []
        tot_count = sum(overall_quality_count.values())
        tot_wt = sum(overall_quality_wt.values())
        for key in overall_quality_count.keys():
            if tot_count == 0:
                count_percent = 0.0
            else:
                count_percent = float((db_class_counts[key] / tot_count) * 100)

            if int(tot_wt) == 0:
                wt_percent = 0.0
            else:
                wt_percent = float((overall_quality_wt[key] / tot_wt) * 100)

            quality_param_name = db_classname_map[key]
            overall_quality_data.append({"quality_parameter_id": key,
                                         "quality_parameter_name": db_classname_map[key],
                                         "count": overall_quality_count[key],
                                         "count_percentage": count_percent,
                                         "weight": overall_quality_wt[key],
                                         "weight_percentage": wt_percent})

        if combined_data is None or combined_data == 0:
            combined_data = {}

        # Calculate the range using length list
        range_status, overall_range_data = range_model(overall_length_sizes_list)

        # Calculating the size and uniformize using length list
        su_status, overall_su_data = size_uniformity_model(overall_length_sizes_list)

        # Club the ranges
        overall_clubbed_range_data = club_range(overall_range_data)

        # Calculate the max length, min length, overall average length
        combined_data["overall_max_length"], combined_data["overall_min_length"], combined_data[
            "overall_avg_lenght"] = find_length_min_max_avg(overall_size_data)

        # Calculate the max width, min width, overall average width
        combined_data["overall_max_width"], combined_data["overall_min_width"], combined_data[
            "overall_avg_width"] = find_width_min_max_avg(overall_size_data)

        combined_data["overall_quality_data"] = overall_quality_data
        combined_data["combined_actual_image_weight"] = combined_actual_total_weight
        combined_data["overall_total_count"] = tot_count

        combined_data["overall_size_data"] = overall_size_data
        combined_data["overall_range_data"] = overall_clubbed_range_data
        combined_data["overall_size_uniformity_data"] = overall_su_data

        image_wise_data = json.dumps(image_wise_data, indent=1)
        image_wise_data = json.loads(image_wise_data)
        image_wise_data = str(image_wise_data).replace("'", '"')

        combined_data = json.dumps(combined_data, indent=1)
        combined_data = json.loads(combined_data)
        combined_data = str(combined_data).replace("'", '"')
        logging.info(f'combined_data - {combined_data}')
        logging.info(f'image_wise_data - {image_wise_data}')

        update_quality_data(ticket_id, image_wise_data, combined_data)
        mydb.close()
        end = t.time()
        total_time = end - start
        logging.info(f"Total Time taken by Quality model: {total_time:.2f} sec")
        if int(invalid_wt_check) == 1:
            return {"message": "Invalid weight for any one of the images.. : " + str(ticket_id), "status": "1"}
        else:
            return {"message": "Quality process done for the ticket id : " + str(ticket_id), "status": "1"}
    else:
        return {"message": "Invalid request method 'GET'", "status": "1"}


@app.route('/quality/test_quality_analysis_app', methods=['GET', 'POST'])
def test():
    return 'The quality_analysis_app is up and running. Send a POST request'


if __name__ == '__main__':
    app.run(port=port_num)
