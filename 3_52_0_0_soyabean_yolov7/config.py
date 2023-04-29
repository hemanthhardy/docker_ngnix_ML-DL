image_server = {'uat': 'https://igrade-testing.waycool.in/',
                'live': 'https://igrade.waycool.in/'}

slice_image_dir = 'sliced_output_images/'

input_dir_path = 'input_images/'

output_merged_image_path = "output_images/"

commodity_specific_model_path = "3_52_0_0/"
model_file_name = "best.pt"
class_file_name = "class_names.txt"
port_num = 9002

class_path = commodity_specific_model_path + class_file_name
inference_graph_path = commodity_specific_model_path

classid_map_file = 'class_names.txt'  # create own file and add manually
commodity_details_file = 'commodity_details.txt'  # create own and add manually
db_classname_map_file = 'db_classname_map.txt'
default_pixel_per_metrix = 15  # Default Pixel Per Metrics Value
image_types = {'quick_image': 1,
               'original': 2,
               'good': 3,
               'semi_processable': 4,
               'unused': 5}
image_types_list = ['original', 'good', 'semi_processable', 'unused']