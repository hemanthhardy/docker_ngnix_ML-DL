import time
from config import *
import os

def try_uploading(image_path):
    print(image_path)
    try:
        myurl = ''
        form_data = {"ticket_id":ticket_id,"image_id":image_id}
        files = {'quality_output':open(image_path,'rb')}
        getdata = requests.post(myurl, form_data, files=files)
        data = json.loads(getdata.text)
        if data['status'] == 1:
            return True
        else:
            return False
        
    except Exception as e:
        print('[Error] : ',e)
        return False
def run():
    while(True):
        print('uploading')
        for j in os.listdir(output_images_path):
            attempts = int(j.split('.')[0].split('_')[-1])
            print(attempts)
            if attempts < max_attempts:
                image = '_'.join(j.split('_')[:-1]) + '_{}.jpg'.format(str(attempts+1))
                image_path = output_images_path + image
                os.rename(output_images_path+j, image_path)
                if try_uploading(image_path):
                    os.remove(image_path)
            else:
                image = '_'.join(j.split('_')[:-1]) + '_{}.jpg'.format(str(attempts+1))
                image_path = output_images_path + image
                os.rename(output_images_path+j, image_path)
                
                if attempts % 20 == 0 and try_uploading(image_path):
                    os.remove(image_path)
                
        time.sleep(1)


if __name__ == '__main__':
    run()
