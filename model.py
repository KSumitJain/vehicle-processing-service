from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone

db = SQLAlchemy()


class PostANPRDetailsModel(db.Model):
    __tablename__ = 'anpr_details'
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, primary_key=False)
    name = db.Column(db.TEXT, primary_key=False)
    license_plate_number = db.Column(db.TEXT, primary_key=False)
    vehicle_file_name = db.Column(db.TEXT, primary_key=False)
    timestamp = db.Column(db.TEXT, primary_key=False)
    created_at = db.Column(db.Date, default=datetime.now(timezone('Asia/Kolkata')).date())

    def __init__(self, camera_id, name, license_plate_number, vehicle_file_name, timestamp):
        self.name = name
        self.camera_id = camera_id
        self.license_plate_number = license_plate_number
        self.vehicle_file_name = vehicle_file_name
        self.timestamp = timestamp


class PostFaceDetectionDetailsModel(db.Model):
    __tablename__ = 'face_details'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.TEXT, primary_key=False)
    face_file_name = db.Column(db.TEXT, primary_key=False)
    timestamp = db.Column(db.TEXT, primary_key=False)
    created_at = db.Column(db.Date, default=datetime.now(timezone('Asia/Kolkata')).date())

    def __init__(self, name, face_file_name, timestamp):
        self.name = name
        self.face_file_name = face_file_name
        self.timestamp = timestamp
