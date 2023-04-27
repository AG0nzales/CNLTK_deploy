# from turtle import title
from flask import Flask, redirect, request, render_template
import os
from joblib import load
# from pos_tagger import predict_POS_model
from CNLTK import preprocessing, pos_tagger, ner
# import string 

app = Flask(__name__)
# set file directory path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))    


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/api", methods=["GET", "POST"])
def api():
    return render_template('apiPage.html')

@app.route("/data", methods=["GET", "POST"])
def data():
    return render_template('dataPage.html')

@app.route("/method", methods=["GET", "POST"])
def method():
    input = [request.form.get("pos")]
    print(input)
    converted_input = str(input[0])
    sentz, tagz = pos_tagger.predict_POS_model(converted_input)
    sentz2, tagz2 = ner.predict_NER_model(converted_input)

    # for word in sentz:
    #     new_sentz += word + ' '
    
    new_sentz = ''
    for i, word in enumerate(sentz, 1):
        new_sentz += "[{}] {} ".format(i, word)
        if i <= len(sentz) - 1:
            new_sentz += ' ' 
        
    
    # for i, word in enumerate(tagz):
    #     new_tagz += word
    #     if i < len(tagz) - 1:
    #         new_tagz += ' - '
    
    new_tagz = ''
    for i, word in enumerate(tagz, 1):
        new_tagz += "[{}] {} ".format(i, word)
        if i <= len(tagz) - 1:
            new_tagz += ' ' 
    
    
    # for word in sentz2:
    #     new_sentz2 += word + ' '
        
    new_sentz2 = ''
    for i, word in enumerate(sentz2, 1):
        new_sentz2 += "[{}] {} ".format(i, word)
        if i <= len(sentz2) - 1:
            new_sentz2 += ' '
    
    
    # for i, word in enumerate(tagz2):
    #     new_tagz2 += word
    #     if i < len(tagz2) - 1:
    #         new_tagz2 += '  ~  '
            
    new_tagz2 = ''
    for i, word in enumerate(tagz2, 1):
        new_tagz2 += "[{}] {} ".format(i, word)
        if i <= len(tagz2) - 1:
            new_tagz2 += ' '


    preproc = preprocessing._POS(converted_input)
    preprocess = preproc[0]
    
    new_proc = ''
    for word in preprocess:
        new_proc += word + ' '
    
    return render_template('methods.html', title='CNLTK Methods', preproc = new_proc, sentz = new_sentz, tagz=new_tagz, sentz2 = new_sentz2, tagz2 = new_tagz2)
    # return render_template('methods.html', title='CNLTK Methods')





if __name__ == '__main__':
    # app.run(debug=True)
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
