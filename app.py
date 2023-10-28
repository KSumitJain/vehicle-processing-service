from flask import Flask, request, jsonify, current_app
from flask_cors import CORS, cross_origin
from app_utils import convert_data_for_mysql, get_data_from_mysql, convert_numpy_array_from_blob
from model import db
import os
import sys
import threading
import logging
import argparse
import torch
yolov7_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML_Model'))
sys.path.insert(0, yolov7_path)

from detect import detect

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
app.secret_key = 'Please Connect to MySQL'
app.logger.addHandler(handler)
app.config.from_pyfile('config.py')
db.init_app(app)

if not os.path.exists(app.config['DB_NAME']):
    with app.app_context():
        db.create_all()

if not os.path.exists('face_imgs'):
    os.makedirs('face_imgs')
# Match Text From The Input

def detect_from_video(file_path, camera_id, weights, is_lpr):
    file_path_no_extension = file_path[:-4]
    file_path_array = file_path_no_extension.split('/')
    file_name = file_path_array[-1]
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
    parser.add_argument('--count', type=int, default=len(os.listdir('face_imgs')), help='count value for face count')
    parser.add_argument('--filename', type=str, default=file_name, help='file name to save as folder')
    parser.add_argument('--camera-id', type=int, default=camera_id, help='camera id')
    opt = parser.parse_args()
    with torch.no_grad():
        resp = detect(opt, app.logger)
        mysql_data = convert_data_for_mysql([resp, camera_id], is_lpr)
        with app.app_context():
            db.session.bulk_save_objects(mysql_data)
            db.session.commit()
    return resp


@app.route("/upload-video-lpr", methods=["POST"])
@cross_origin(supports_credentials=True)
def match_details():
    if 'cameraId' not in request.args:
        return 'No Camera ID Mentioned'
    if 'sourcePath' not in request.args:
        return 'No Source Path Mentioned'
    if 'timestamp' not in request.args:
        request.args['timestamp'] = '2023-09-07 00:00:00'
        #return 'No Time Stamp Mentioned'
    thread = threading.Thread(target=detect_from_video, args=(request.args['sourcePath'], request.args['cameraId'], request.args['timestamp'], app.config['LPR_WEIGHT_FILE'], True))
    thread.start()
    response = jsonify({"status": 'File Uploaded Successfully'})
    response.status_code = 200
    return response

'''
@app.route("/upload-video-fda", methods=["POST"])
@cross_origin(supports_credentials=True)
def get_faces():
    if 'sourcePath' not in request.args:
        return 'No Source Path Metioned'
    thread = threading.Thread(target=detect_from_video, args=(request.args['sourcePath'], app.config['LPR_WEIGHT_FILE'], False))
    thread.start()
    response = jsonify({"status": 'File Uploaded Successfully'})
    response.status_code = 200
    return response
'''

@app.route("/check-blob-data", methods=['GET'])
@cross_origin(supports_credentials=True)
def check_blob_value():
    if 'id' not in request.args:
        return 'No Id Passed'
    row = get_data_from_mysql(request.args['id'], False)
    if row:
        reconstructed_values = convert_numpy_array_from_blob(row.face_binary)
        response = jsonify(
            {"id": row.id, "videoName": row.name, "faceArray": reconstructed_values, "timestamp": row.timestamp})
        response.status_code = 200
    else:
        response = jsonify('Record with that id does not exists in DB')
        response.status_code = 204
    return response


if __name__ == "__main__":
    app.run(debug=False, host='localhost')
