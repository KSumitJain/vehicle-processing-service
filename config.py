DB_NAME = 'trackingdata'
DB_HOST = 'localhost'
DB_USERNAME = 'root'
DB_PASSWORD = 'root'
SQLALCHEMY_DATABASE_URI = 'mysql://'+DB_USERNAME+':'+DB_PASSWORD+'@'+DB_HOST+'/'+DB_NAME
LPR_WEIGHT_FILE = 'ML_Model/LPR_Yolo_Tiny.pt'
FACE_DET_WEIGHT_FILE = 'ML_Model/Face_Det_Yolo_Tiny.pt'
FRAME_LOCATION = '../../frame-images'
