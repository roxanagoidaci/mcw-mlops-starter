import os
import json
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import onnxruntime





def init():
    global model
    global inputs_dc, prediction_dc
    
    try:
        model_name = 'MODEL-NAME' # Placeholder model name
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name = model_name)
        print('Loading model from: ', model_path)
        # Load the ONNX model
        model = onnxruntime.InferenceSession(model_path)
        print('Model loaded...')

        inputs_dc = ModelDataCollector("model_telemetry", designation="inputs")
        prediction_dc = ModelDataCollector("model_telemetry", designation="predictions", feature_names=["prediction"])


    except Exception as e:
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    import time
    try:
        print("Received input: ", raw_data)
        
        inputs = np.array(json.loads(raw_data)).astype(np.float32)
       
        results = model.run(None, {model.get_inputs()[0].name:inputs})[0]
        results = results[0][0].item()

        inputs_dc.collect(inputs) #this call is saving our input data into Azure Blob
        prediction_dc.collect(results) #this call is saving our output data into Azure Blob

        print("Prediction created " + time.strftime("%H:%M:%S"))
        
        results = results.tolist()
        return json.dumps(results)
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error

