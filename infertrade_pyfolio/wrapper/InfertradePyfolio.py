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


from pyfolio import tears, plotting, utils
from infertrade_pyfolio.infer_utils import customs

# Cumulative to non-cumulative return
def noncumulative(returns):
    return (returns.diff().fillna(0))

# Return tear sheet
def infertrade_return_tear_sheet(returns, bench_returns=None, return_fig=True,
                                 fulltear = False, notebook=False, pdf=True):
    returns = noncumulative(returns)
    if bench_returns is not None:
        bench_returns = noncumulative(bench_returns)
    plotting.show_perf_stats = customs.cust_show_perf_stats
    plotting.show_worst_drawdown_periods = customs.cust_show_worst_drawdown_periods
    fig = tears.create_returns_tear_sheet(returns=returns, benchmark_rets=bench_returns, return_fig=return_fig)
    if fulltear:
        return fig
    else:
        customs._isJupyterNB(NoteBook=notebook, fig_instance=[fig], plot_name="Return tear sheet", pdf=pdf)

# Intresting time tear sheet
def infertrade_intresting_time_tear_sheet(returns, bench_returns=None, return_fig=True,
                                          fulltear = False, notebook=False, pdf=True):
    returns = noncumulative(returns)
    if bench_returns is not None:
        bench_returns = noncumulative(bench_returns)
    utils.print_table = customs.cust_print_table
    fig = tears.create_interesting_times_tear_sheet(returns=returns, benchmark_rets=bench_returns, return_fig=return_fig)
    if fulltear:
        return fig
    else:
        customs._isJupyterNB(NoteBook=notebook, fig_instance=[fig], plot_name="Intresting time tear sheet", pdf=pdf)

# Full tear sheet
def infertrade_full_tear_sheet(returns, bench_returns=None, return_fig=True,
                               fulltear = True, notebook=False, pdf=True):
    fig1 = infertrade_return_tear_sheet(returns=returns, bench_returns=bench_returns, return_fig=return_fig,
                                        fulltear = fulltear, notebook = notebook, pdf=pdf)
    fig2 = infertrade_intresting_time_tear_sheet(returns=returns, bench_returns=bench_returns, return_fig=return_fig,
                                                 fulltear=fulltear, notebook=notebook, pdf=pdf)
    customs._isJupyterNB(NoteBook=notebook, fig_instance=[fig1, fig2], plot_name="Full tear sheet", pdf=pdf)
