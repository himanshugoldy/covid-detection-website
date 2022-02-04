from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)

dic = {0 : 'Covid! Stay Indoors and Wear Mask', 1 : 'Normal! Be Safe'}

cm = load_model('covidModel.hdf5')

cm.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1,224,224,3)
    p = cm.predict(i)
    y_pred = []
    y_pred.append(p[0,0])
    y_pred = np.array(y_pred)>0.5
    y_pred = y_pred.astype('int32')
    return dic[y_pred[0]]
    #return "Covid"
    # Executed if error in the
    # try block
    #label = np.array(p)[0,0] > .5
    #print('----')
    #print(label)
    

def predict_label2(img_path,cm):
    return 'Error in the model'

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Covid Prediction Model"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename	
        img.save(img_path)
        p = predict_label(img_path)
        #try:
         #   p = predict_label(img_path)
       # except Exception as e:
           # print(e) 
    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
    app.run(debug = True)
    #img = request.files['my_image']
    img_path = "C:/Users/user/OneDrive/Desktop/Ml Model/static/1.jpg"	
    #img.save(img_path)
    p = predict_label(img_path)
    print(p)
    # except Exception as e:
    #     print(e) 