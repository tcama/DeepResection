from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os, sys
sys.path.insert(1, '../scripts')
from calculate_resected_volumes import calc_resec_vol
from generate_mask import gen_mask, dice_coeff, dice_loss, adjust_sizes
from jinja2 import Template
from flask import Markup
import sqlite3

from tensorflow.compat.v1.keras.models import Model, load_model
import nibabel as nib
import numpy as np
import time
import pickle

app = Flask(__name__)
app.secret_key = b'sdfpow23'
model = None

print(app.root_path)

def load_unet_model():
    global model
    model = load_model('../analysis/model_inception.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vol', methods=['GET', 'POST'])
def vol():
    if request.method == 'POST':
        tic = time.perf_counter()

        # [img, mask, atl, atl_map] = request.files.getlist("file")
        [img, mask] = request.files.getlist("file")
        atlas = request.form['atlas']

        print([img, mask])

        for uploaded_file in [img, mask]:
            if uploaded_file.filename != '':
                uploaded_file.save(os.path.join("static", uploaded_file.filename))

        if atlas == "AAL":
            atlas_nii = "tmp/atlas2post_AAL116_origin_MNI_T1.nii"
            atlas_txt = "tmp/AAL116.txt"

        df, imgs = calc_resec_vol(
            os.path.join("static", img.filename),
            os.path.join("static", mask.filename),
            # os.path.join("static", atl.filename),
            # os.path.join("static", atl_map.filename),
            atlas_nii,
            atlas_txt,
            os.path.join("static", ""),
        )

        df.sort_values(by="Remaining (%)", inplace=True)

        with open('temp_reportdata.pickle', 'wb') as handle:
            pickle.dump({'df': df, 'imgs': imgs}, handle, protocol=pickle.HIGHEST_PROTOCOL)


        table_html = df.to_html(
                table_id='report',
                float_format=lambda x: '{0:.2f}'.format(x),
                index=False,
                justify="center"
                )

        toc = time.perf_counter()

        return redirect(url_for('report', vol_table=Markup(table_html), post_op_path=img.filename, mask_path=mask.filename))
    return render_template('calculate_vol.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        try:
            rating = request.form['rating']
            comments = request.form['comments']

            print(rating, comments)

            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                print('test')

                cur.execute("INSERT INTO feedback (date_time,rating,comments) VALUES (datetime('now'),?,?)",(rating,comments) )
                
                con.commit()
                msg = "Record successfully added"
                print(msg)
        except Exception as e:
            print(e)
            con.rollback()
            msg = "error in insert operation"
        
        finally:
            return redirect(url_for("index"))
            con.close()

@app.route('/report', methods=['GET', 'POST'])
def report():
    vol_table = request.args.get('vol_table')
    vol_imgs = request.args.get('vol_imgs')
    post_op_path = request.args.get('post_op_path')
    mask_path = request.args.get('mask_path')

    print(os.listdir("tmp"))
    rendered_template = render_template('report.html', vol_table=Markup(vol_table), post_op_path=post_op_path, mask_path=mask_path)

    if request.method == 'POST':
        try:
            # rendered_template_enc = rendered_template.encode('utf-8') 
            css = ['./static/css/style.css', './static/css/papaya.css']
            pdf = pdfkit.from_string(rendered_template, 'report.pdf', css=css)
        except Exception as e:
            print(e)
    return rendered_template

@app.route('/mask', methods=['GET', 'POST'])
def mask():
    if request.method == 'POST':
        tic = time.perf_counter()
        patient_id = request.form['id']
        atlas = request.form['atlas']
        isContinuous = 'continuous' in request.form
        postop = request.files['file']
        postop.save(os.path.join("static", postop.filename))

        if atlas == "AAL":
            atlas_nii = "tmp/atlas2post_AAL116_origin_MNI_T1.nii"
            atlas_txt = "tmp/AAL116.txt"

        mask_name="{}_predicted_mask.nii.gz".format(patient_id)

        print(mask_name)

        gen_mask(
            os.path.join("static", postop.filename),
            "static",
            mask_name,
            isContinuous
        )

        df, imgs = calc_resec_vol(
            os.path.join("static", postop.filename),
            os.path.join("static", mask_name),
            atlas_nii,
            atlas_txt,
            os.path.join("static", ""),
        )


        df.sort_values(by="Remaining (%)", inplace=True)

        table_html = df.to_html(
                table_id='report',
                float_format=lambda x: '{0:.2f}'.format(x),
                index=False,
                justify="center"
                )
        toc = time.perf_counter()
        print("Time elapsed: {:.4f}".format(toc - tic))
        
        return redirect(url_for('report', vol_table=Markup(table_html), post_op_path=postop.filename, mask_path=mask_name))
    return render_template('generate_mask.html')    

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # load_unet_model()
    app.run(debug=True)