from flask import Flask
from extract_cnn_vgg16_keras import VGGNet
from flask_restful import reqparse, abort, Api, Resource
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import cv2
import os
import h5py
from query_api import get_image_search

app = Flask(__name__)
api = Api(app)

RE = {
    "get" : {"Doc" : "Unipower Image Search System Restful Api"},
    "WARN" : {"WARNING" : "Null"}
}

parser = reqparse.RequestParser()
parser.add_argument('image', type=FileStorage, location='files')

class ImageSearch(Resource):
    def get(self):
        return RE['get'], 200

    def post(self):
        args = parser.parse_args()
        
        # store image file
        im_file = args.get('image')
        im_name = secure_filename(im_file.filename)
        im_file.save(os.path.join('image/', im_name))
        
        im_file = os.path.join('image/',im_name)
        
        result_dict = get_image_search(im_file)
        if result_dict:
            return result_dict, 201
        else:
            return RE['WARN'], 404 

api.add_resource(ImageSearch, '/imagesearch')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9393, debug=False)
    #app.run()
