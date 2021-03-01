import keras
from flask import Flask
from flask import request
from flask import make_response
# ## 1. Import libraries and load data
#packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load data
#@app.route('/genres/train', methods=['POST','PUT']) 
#def endpoint_train():
#    filename = request.files['csv'].filename
#    train= pd.read_csv(filename)
    
#    return train
app = Flask(__name__)
@app.route('/genres/predict', methods=['POST','PUT'])
def endpoint_test():
    # filename = request.files['csv'].filename
    # test = pd.read_csv(filename)
    # outputFile = createSubmission(test)

    csv = 'foo,bar,baz\nhai,bai,crai\n'
    response = make_response(csv)
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    response.headers["Content-type"] = "text/csv"

    return response

if __name__ == "__main__":
    app.debug=True
    app.run()

