watershed_model_map = {"58": "wheat_bg",
                       "49": "rice_bg",
                       "40": "maize_bg",
                       "29": "green_gram_bg",
                       "32": "green_gram_splits_bg",
                       "50": "sesame_seeds_wg",
                       "4": "niger_seeds_wg",
                       "17": "black_pepper_wg",
                       "33": "kidney_beans_wg",
                       "3": "ragi_wg",
                       "71": "red_gram_white_bg",
                       "18": "cardamom",
                       "68": "pista",
                       "46": "raisin"}  # commodity specific background
black_bg_map = ["48", "31", "62", "23", "14", "6", "13", "66", "47"]  # list of commodities with only black bg
other_small_comm_map = ["7", "52", "43", "16"]
gka_bg_map = ["69", "70", "10"]
white_bg_map = ["15", "41"]

input_images_path = "input_images/"
anomaly_model_path = 'anomaly/anomaly_detector.model'
range_clubbing_percent = 5.0
start_range = 0.0
end_range = 35.0
image_server = {'uat': 'https://igrade-testing.waycool.in/',
                'live': 'https://igrade.waycool.in/'}
# anomaly_model_path = 'anomaly/anomaly_detector.model'
canny_output_path = 'output_images/'
watershed_output_path = 'output_images/'
upload_image_url = 'https://igrade-testing.waycool.in/output_images/upload_output_images/'
port_num = 8080

