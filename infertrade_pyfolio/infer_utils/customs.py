# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by: Melroy Pereira and Nikola Rokvic
# Created date: 9 Dec 2021
# Copyright 2021 InferStat Ltd


import datetime
import string
import matplotlib
from PIL import Image
from pyfolio import plotting
import matplotlib.pyplot as plt

# Override function for show perf stats from plotting
def cust_show_perf_stats(returns,benchmark_rets=None,positions=None,transactions=None,
                        turnover_denom=None, bootstrap=None, live_start_date=None,
                        header_rows=None):
    pass

# Override function for show worst drawdown periods from plotting
def cust_show_worst_drawdown_periods(returns):
    pass

# Override function for print table from utils
def cust_print_table(table, name=None, float_format=None,
                     formatters=None, header_rows=None):
    pass

# output parsing function
def tear_sheet_output(figlist, plot_name):
    # file time
    stringtime = str(datetime.datetime.now()).replace("-", "")
    stringtime = stringtime.translate(str.maketrans('', '', string.punctuation)).replace(" ", '')
    # looping the figure instances to save the plot figure
    if len(figlist)==1:
        return (figlist[0].savefig(plot_name+stringtime+".pdf", format="pdf"))
    else:
        for ix, im_ in enumerate(figlist):
            im_.savefig("image"+str(ix)+".jpg")
        # list for saving the figure to append in pdf
        imis = []
        for i in range(len(figlist)):
            # First image is used as primary page to append other so we dont append
            if (i==0):
                im0 = Image.open("image"+str(i)+".jpg")
            else:
                imis.append(Image.open("image"+str(i)+".jpg"))
        # pdf output
        pdf = plot_name+stringtime+".pdf"
        return(im0.save(pdf, "PDF", resolution=100.0, save_all=True, append_images=imis))


# Is jupyter notebook function
def _isJupyterNB(NoteBook, fig_instance, plot_name, pdf):
    if ((NoteBook==False) and (pdf==True)):   #IDE
        return (tear_sheet_output(fig_instance, plot_name=plot_name))
    elif ((NoteBook == True) and (pdf == True)):
        return (tear_sheet_output(fig_instance, plot_name=plot_name))