import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template


def vol_report(HTML_DIR, df, fig):
    df.sort_values(by="Remaining (%)", inplace=True)

    with open('../views/template.html') as f:
        t = Template(f.read())

    vals = {
        "vol_table": df.to_html(
            table_id='report',
            float_format=lambda x: '{0:.2f}'.format(x),
            index=False,
            justify="center"
            ),
        "vol_imgs": "\n <img src=\"" + "resection_views.png" + "\" align=\"top right\"/>"
    }

    with open(HTML_DIR, 'w') as f:
        f.write(t.render(vals))
    
    return(HTML_DIR)