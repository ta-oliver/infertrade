"""
Functions to generate a dictionary wrapping ta's methods.

Copyright 2021 InferStat Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created by: Nikolay Alemasov
Created date: 25/03/2021
"""
import inspect, json, warnings
from ta import momentum, trend, volatility, volume, others

def print_dict(export: dict):
	"""Formats and outputs the generated dictionary into console."""
	
	def p_(*args, level = 0, ident = 4):
		"""Printing function. Now outputs into console."""
		if level > 0:
			print(" "*(level*ident-1), *args)
		else:
			print(*args)
			
	p_("ta_export_signals = {")

	for k, v in export.items():
		p_("\"%s\": {"%k, level = 1)
		
		for key, val in v.items():
			if key == 'class':
				msg = val.__name__
			else:
				msg = json.dumps(val)

			p_("\"%s\": %s,"%(key, msg), level = 2)

		p_("},", level = 1)
	
	p_("}")
	

def inspect_ta_module(module, series_params = ['open', 'high', 'low', 'close', 'volume'], include_bools = False) -> dict:
	"""Inspects the ta module provided. Outputs a dictionary based on the classes found and their methods."""
	
	def _gen_SMAIndicator_variants(desc: dict, windows = [20, 50, 200]) -> dict:
		"""Generates individual variants in the case of SMAIndicator since its single parameter "window" has no default values."""
		
		assert len(desc['parameters'].keys()) == 1
		assert list(desc['parameters'].keys())[0] == 'window'
		assert list(desc['parameters'].values())[0] is None

		res = {}

		for w in windows:
			d = desc.copy()
			d.update({'parameters': {'window': w}})
			res.update({'SMA%d'%w: d})
			
		return res
	
	ta_export_signals = {}
	
	for n, c in inspect.getmembers(module, inspect.isclass):
		if n == 'IndicatorMixin':
			continue
			
		csig = inspect.signature(c.__init__)
		
		series = []
		params = {}
		
		for p, s in csig.parameters.items():
			if p == 'self' or p in ['fillna']:
				continue
				
			if s.annotation == bool and not include_bools:
				continue
				
			if p in series_params:
				series.append(p)
				continue
				
			default = s.default
				
			if s.default is inspect.Parameter.empty:
				default = None
				warnings.warn('Default value is empty for %s'%('.'.join([cls.__module__, n, p])))
			
			params.update({p: default})
		
		methods = []
		
		for m, f in inspect.getmembers(c, inspect.isfunction):
			if m.startswith('_'):
				continue
				
			sig = inspect.signature(f)
			
			for i, p in enumerate(sig.parameters):
				if i > 0 or p != 'self':
					assert()
			
			methods.append(m)
			
		assert(len(methods) > 0)
		assert(len(series) > 0)

		cls = c
	
		for m in methods:
			desc = {'class': cls, 'module': cls.__module__, 'function_names': m, 'parameters': params, 'series': series}
			
			assert(not m in ta_export_signals.keys())
			
			if cls.__name__ == 'SMAIndicator':
				ta_export_signals.update(_gen_SMAIndicator_variants(desc))
			else:
				ta_export_signals.update({m: desc})

	return ta_export_signals
	
if __name__ == "__main__":
	signals = {}
	
	for m in [momentum, trend, volatility, volume, others]:
		signals.update(inspect_ta_module(m))

	print_dict(signals)
