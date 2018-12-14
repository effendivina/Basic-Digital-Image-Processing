import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    uploaded_file = request.files.getlist("file")
    for file in uploaded_file:
        print(file)
        if file.filename == "":
            return render_template("no_img.html", file_path="img/no_image_selected.gif")

        filename = "temp_img.jpg"
        destination = "/".join([target, filename])
        print(destination)
        file.save("static/img/temp_img.jpg")
        copyfile("static/img/temp_img.jpg","static/img/normal_img.jpg")

        return render_template("uploaded.html", file_path="img/temp_img.jpg")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    image_processing.normal()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/moveleft", methods=["POST"])
@nocache
def moveleft():
    image_processing.moveleft()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/moveright", methods=["POST"])
@nocache
def moveright():
    image_processing.moveright()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/moveup", methods=["POST"])
@nocache
def moveup():
    image_processing.moveup()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/movedown", methods=["POST"])
@nocache
def movedown():
    image_processing.movedown()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/brightplus", methods=["POST"])
@nocache
def brightplus():
    image_processing.brightplus()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/brightsubs", methods=["POST"])
@nocache
def brightsubs():
    image_processing.brightsubs()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/brightmulti", methods=["POST"])
@nocache
def brightmulti():
    image_processing.brightmulti()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/brightdiv", methods=["POST"])
@nocache
def brightdiv():
    image_processing.brightdiv()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/view_histogram", methods=["POST"])
@nocache
def view_histogram():
    image_processing.view_histogram()
    return render_template("uploaded_viewhist.html", file_paths=["img/temp_img_red_histogram.jpg", "img/temp_img_green_histogram.jpg", "img/temp_img_blue_histogram.jpg"])

@app.route("/hist_eq", methods=["POST"])
@nocache
def hist_eq(): 
    image_processing.hist_eq()
    return render_template("uploaded_histeq.html", file_path="img/temp_img.jpg")

@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/sharp", methods=["POST"])
@nocache
def sharp():
    image_processing.sharp()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/edge", methods=["POST"])
@nocache
def edge():
    image_processing.edge()
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/input_threshold", methods=["POST"])
@nocache
def input_threshold():
    return render_template("uploaded_threshold.html", file_path="img/temp_img.jpg")

@app.route("/threshold", methods=["POST"])
@nocache
def threshold():
    red = [int(request.form['redbawah']),int(request.form['redatas'])]
    green = [int(request.form['greenbawah']),int(request.form['greenatas'])]
    blue = [int(request.form['bluebawah']),int(request.form['blueatas'])]
    segmen_color = [int(request.form['red']),int(request.form['green']),int(request.form['blue'])]
    
    image_processing.threshold(red, green, blue, segmen_color)
    # except:
    #     return render_template("uploaded_threshold.html", file_path="img/temp_img.jpeg", alert="Matrix must filled all by integers")
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

@app.route("/input_region", methods=["POST"])
@nocache
def input_region():
    return render_template("uploaded_regiongrowth.html", file_path="img/temp_img.jpg")

@app.route("/region_growth", methods=["POST"])
@nocache
def region_growth():
    seed = (int(request.form['seed_x']),int(request.form['seed_y']))
    threshold_seed = int(request.form['threshold_seed'])
    # segmen_color = [int(request.form['red']),int(request.form['green']),int(request.form['blue'])]
    # segmen_color = int(request.form['red'])
    
    image_processing.region_growth(seed, threshold_seed)
    return render_template("uploaded.html", file_path="img/temp_img.jpg")

if __name__ == '__main__':
    app.run(debug=True)
