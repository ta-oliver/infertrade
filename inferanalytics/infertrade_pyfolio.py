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
# Created date: 5 Dec 2021
# Copyright 2021 InferStat Ltd


# Importing required libraries

import pandas as pd
import datetime
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from inferanalytics import tears

# Is jupyter notebook function
def _isJupyterNB(NoteBook, fig_instance, plot_name, pdf):
    if ((NoteBook==False) and (pdf==True)):   #IDE
        return (tear_sheet_output(fig_instance, plot_name=plot_name))
    elif ((NoteBook == True) and (pdf == True)):
        return (tear_sheet_output(fig_instance, plot_name=plot_name))

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



# return tear sheet
def infertrade_return_tear_sheet(returns, benchmark_rets=None, return_fig=True,
                                 plot_name="return_tear_sheet",full_sheet=False, NoteBook=False, pdf=True):
    fig_instance = tears.create_returns_tear_sheet(returns=returns, positions=None, transactions=None, live_start_date=None,
                                cone_std=(1.0, 1.5, 2.0), benchmark_rets=benchmark_rets, bootstrap=False,
                                turnover_denom="AGB", header_rows=None, set_context=None, return_fig=return_fig)
    if full_sheet:
        return fig_instance
    else:
        _isJupyterNB(NoteBook, [fig_instance], plot_name, pdf)


# interesting times tear sheet
def infertrade_interesting_times_tear_sheet(returns, benchmark_rets=None, return_fig=True, plot_name = "Intresting_times_tear_sheet",
                                            full_sheet=False, NoteBook=False, pdf=True):
    fig_instance = tears.create_interesting_times_tear_sheet(returns=returns, benchmark_rets=benchmark_rets, periods=None,
                                                   legend_loc="best", return_fig=return_fig,)
    if full_sheet:
        return  fig_instance
    else:
        _isJupyterNB(NoteBook, [fig_instance], plot_name, pdf)

# full tear sheet
def infertrade_full_tear_sheet(returns, benchmark_rets=None, plot_name = "Full_tear_sheet", NoteBook=False, pdf=True):

    fig_instance = []
    return_fig_instance = infertrade_return_tear_sheet(returns=returns,benchmark_rets=benchmark_rets,
                                 return_fig=True, plot_name=plot_name+"return_sheet",
                                full_sheet=True, NoteBook=NoteBook, pdf=pdf)
    fig_instance.append(return_fig_instance)

    intrest_time_fig_instance = infertrade_interesting_times_tear_sheet(returns=returns, benchmark_rets=benchmark_rets,return_fig=True,
                                                                        plot_name=plot_name+"intresting_times_sheet", full_sheet=True,
                                                                        NoteBook=NoteBook, pdf=pdf)

    fig_instance.append(intrest_time_fig_instance)

    _isJupyterNB(NoteBook, fig_instance, plot_name, pdf)