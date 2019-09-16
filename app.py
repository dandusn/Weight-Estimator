import os
from main import app
from flask import flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from Engine import engine

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('HomePage.html')

@app.route('/', methods=['POST'])
def upload_file():
    allow = False
    f = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            age = request.form['age']
            out = engine.main(filename, age)
            flash("Predicted Weigth: " + str(out) + " Kg")
            full_filename = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            return render_template("Result.html", user_image = full_filename)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)