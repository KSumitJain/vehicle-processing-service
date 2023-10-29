from flask import Flask, request, jsonify, current_app
from flask_cors import CORS, cross_origin
from app_utils import convert_data_for_mysql, get_data_from_mysql, convert_numpy_array_from_blob, make_parser
from model import db
import os
import sys
import threading
import logging
import torch
import datetime
from pytz import timezone
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

if not os.path.exists(app.config['FRAME_LOCATION']):
    os.makedirs(app.config['FRAME_LOCATION'])
# Match Text From The Input

def detect_from_video(file_path, camera_id, timestamp, weights, is_lpr):
    file_path_no_extension = file_path[:-4]
    file_path_array = file_path_no_extension.split('/')
    file_name = file_path_array[-1]
    if not os.path.exists(app.config['FRAME_LOCATION']+'/Camera_'+str(camera_id)+'/'+file_name+'/'):
        os.makedirs(app.config['FRAME_LOCATION']+'/Camera_'+str(camera_id)+'/'+file_name+'/')
    parser = make_parser(weights, file_path, is_lpr, file_name, camera_id, timestamp, len(os.listdir(app.config['FRAME_LOCATION']+'/Camera_'+str(camera_id))), app.config['FRAME_LOCATION'])
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
    timestamp = datetime.datetime.now(timezone('Asia/Kolkata'))
    #return 'No Time Stamp Mentioned'
    thread = threading.Thread(target=detect_from_video, args=(request.args['sourcePath'], request.args['cameraId'], timestamp, app.config['LPR_WEIGHT_FILE'], True))
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
