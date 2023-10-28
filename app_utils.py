from model import PostANPRDetailsModel, PostFaceDetectionDetailsModel
import numpy as np
from flask import jsonify

def convert_numpy_array_from_blob(flat_arr):
    arr = flat_arr
    return arr

def convert_data_for_mysql(input_data, is_lpr):
    entries = input_data[1]
    camera_id = input_data[2]
    arr = []
    for item in entries:
        for sub in item:
            arr.append(sub)
    for item in arr:
        item.insert(0, camera_id)
    i = 0
    for item in arr:
        if is_lpr:
            arr[i] = PostANPRDetailsModel(camera_id=item[0],
                                          license_plate_number=item[2][0],
                                          vehicle_file_name=item[2][1],
                                          timestamp=item[1],
                                          )
        else:
            arr[i] = PostFaceDetectionDetailsModel(name=item[0],
                                                   face_file_name=item[2],
                                                   timestamp=item[1],
                                                   )
        i += 1
    print('Array is: ', arr)
    return arr


def get_data_from_mysql(id, is_lpr):
    if is_lpr:
        return PostANPRDetailsModel.query.filter_by(id=int(id)).first()
    return PostFaceDetectionDetailsModel.query.filter_by(id=int(id)).first()
