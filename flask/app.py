from flask import Flask, render_template, request, redirect, url_for
import os, sys

sys.path.insert(1, '../scripts')
from calculate_resected_volumes import calc_resec_vol
from generate_mask import gen_mask
from pre2post import pre2post
from pre2post_deformable import pre2post_deformable
from register_atlas_to_preop import register_atlas_to_preop

from jinja2 import Template
from flask import Markup
import sqlite3
import pdfkit
import time
import base64

app = Flask(__name__)
app.secret_key = b'sdfpow23'

import tensorflow as tf
graph = tf.compat.v1.get_default_graph()

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
            atlas_nii = "tmp/atlas2post_AAL116_origin_MNI_T1old.nii"
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

        table_html = df.to_html(
                table_id='report',
                float_format=lambda x: '{0:.2f}'.format(x),
                index=False,
                justify="center"
                )

        toc = time.perf_counter()

        return redirect(url_for('report', vol_table=Markup(table_html), vol_imgs=imgs, post_op_path=img.filename, mask_path=mask.filename))
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

    HTML_TEMPLATE = '''
        <!doctype html>
        <html lang="en">
        <head>
            <!-- Required meta tags -->
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

            <!-- Bootstrap CSS -->
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

            <title>DeepResection Report</title>
        </head>
        
        <body>


                <!-- Viewer -->
                <div class="background-box container">
                    <div class="page-header">
                        <h1>DeepResection Report</h1>
                    </div>
                
                    <div class="row">
                        <div class="table table-striped">
                            {vol_table}
                        </div>
                    </div>
                    <div class="row">
                        <img src="data:image/png;base64, {vol_imgs}"/>
                    </div>
                </div>

        </body>
        </html>
        '''

    img = base64.b64encode(open("./static/resection_views.png", "rb").read()).decode()

    options = {'enable-local-file-access': None}
    pdf = pdfkit.from_string(HTML_TEMPLATE.format(vol_table=Markup(vol_table), vol_imgs=str(img)), './static/report.pdf', css='./static/css/style.css', options=options)

    print(os.listdir("tmp"))
    if request.method == 'POST':
        try:
            # rendered_template_enc = rendered_template.encode('utf-8') 
            vol_table = request.args.get('vol_table')
            vol_imgs = request.args.get('vol_imgs')
            print("Print Test")
            print("Vol ", vol_table)
            # rendered_html_for_pdf = render_template('report_pdf.html', vol_table=Markup(vol_table), vol_imgs=vol_imgs)

            css = ['./static/css/style.css']
            options = {'enable-local-file-access': None}
            pdf = pdfkit.from_string(HTML_TEMPLATE.format(vol_table=Markup(vol_table), vol_imgs=vol_imgs), 'report.pdf', css=css, options=options)

            return render_template('report.html', vol_table=Markup(vol_table), post_op_path=post_op_path, mask_path=mask_path)
        except Exception as e:
            print(e)
    return render_template('report.html', vol_table=Markup(vol_table), post_op_path=post_op_path, mask_path=mask_path)

@app.route('/mask', methods=['GET', 'POST'])
def mask():
    if request.method == 'POST':
        tic = time.perf_counter()
        patient_id = request.form['id']
        atlas = request.form['atlas']
        isContinuous = 'continuous' in request.form
        # postop = request.files['file']
        
        if len(request.files.getlist("file")) == 2:
            [preop, postop] = request.files.getlist("file")
        elif len(request.files.getlist("file")) == 3:
            [preop, postop, mask] = request.files.getlist("file")
            mask.save(os.path.join("static", mask.filename))

        preop.save(os.path.join("static", preop.filename))
        postop.save(os.path.join("static", postop.filename))
        
        isDeformable = "deformable" in request.form
        # Deprecated    
        # if atlas == "AAL":
        #     atlas_nii = "tmp/atlas2post_AAL116_origin_MNI_T1.nii"
        #     atlas_txt = "tmp/AAL116.txt"

        if atlas == 'DKT':
            atlas_nii = None
            atlas_txt = "tmp/dkt_atlas_mappings.txt"

        mask_name="{}_predicted_mask.nii.gz".format(patient_id)

        output_dir = "static"
        global graph

        # apply an atlas to pre-operative image, register atlas to post-operative image
        print('pre2post')
        print("deformable? {}".format(isDeformable))
        if isDeformable:
            pre2post_deformable(
                patient_id,
                os.path.join(output_dir, preop.filename),
                os.path.join(output_dir, postop.filename),
                output_dir,
                os.path.join(output_dir, mask.filename)
            )

        else:
            pre2post(
                patient_id,
                os.path.join(output_dir, preop.filename),
                os.path.join(output_dir, postop.filename),
                output_dir
            )

        pre2post_fname = "pre2post_{}".format(preop.filename)

        print("register_atlas_to_preop")
        registered_atlas_fname = register_atlas_to_preop(
            patient_id,
            os.path.join(output_dir, pre2post_fname),
            output_dir
        )

        print("gen_mask")
        with graph.as_default():
            gen_mask(
                os.path.join(output_dir, postop.filename),
                output_dir,
                mask_name,
                isContinuous
            )

        print("calc_resec_volume")
        df, imgs = calc_resec_vol(
            os.path.join(output_dir, postop.filename),
            os.path.join(output_dir, mask_name),
            os.path.join(output_dir, registered_atlas_fname),
            atlas_txt,
            output_dir,
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
        
        return redirect(url_for('report', vol_table=Markup(table_html), vol_imgs=imgs, post_op_path=postop.filename, mask_path=mask_name))
    return render_template('generate_mask.html')    

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
