from model import PostANPRDetailsModel, PostFaceDetectionDetailsModel
import argparse
import datetime
def convert_numpy_array_from_blob(flat_arr):
    arr = flat_arr
    return arr

def make_parser(weights, file_path, is_lpr, file_name, camera_id, timestamp, count, frame_location):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=file_path, help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='ML_Model/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--timestamp', type=str, default=[], help='timestamp of the number')
    parser.add_argument('--is-lpr', type=bool, default=is_lpr, help='check if license plate recognition or face det')
    parser.add_argument('--count', type=int, default=count, help='count value for face count')
    parser.add_argument('--filename', type=str, default=file_name, help='file name to save as folder')
    parser.add_argument('--camera-id', type=int, default=camera_id, help='camera id')
    parser.add_argument('--orig-timestamp', type=datetime.datetime, default=timestamp, help='Orig Timestamp of the period')
    parser.add_argument('--frame-location', type=str, default=frame_location)
    return parser

def convert_data_for_mysql(input_data, is_lpr):
    entries = input_data[0]
    camera_id = input_data[1]
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
    return arr


def get_data_from_mysql(id, is_lpr):
    if is_lpr:
        return PostANPRDetailsModel.query.filter_by(id=int(id)).first()
    return PostFaceDetectionDetailsModel.query.filter_by(id=int(id)).first()
