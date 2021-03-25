import re, json
import pandas as pd

import inspect
from ta import momentum, trend, volatility, volume, others

def print_dict_(export):
	
	def p_(*args, level = 0, ident = 4):
	
		if level > 0:
			print(" "*(level*ident-1), *args)
		else:
			print(*args)
			
	def fmt_(x):
		if isinstance(x, str):
			return "\"%s\""%x
			
		if isinstance(x, list):
			return "\"%s\""%x
		
	p_("ta_export_signals = {")

	for k, v in export.items():
		p_("\"%s\": {"%k, level = 1)
		
		for key, val in v.items():
			if key == 'class':
				msg = val.__name__
			else:
#				print(val)
				msg = json.dumps(val)

			p_("\"%s\": %s,"%(key, msg), level = 2)

		p_("},", level = 1)
	
	p_("}")
	

def inspect_(module, series_params = ['open', 'high', 'low', 'close', 'volume']):
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
				
			if p in series_params:
				series.append(p)
				continue
				
			default = s.default
				
			if s.default is inspect.Parameter.empty:
				default = 'to_be_assigned'
				print('[Warning] default is empty for', cls.__module__, n, p)
			
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
		
	
		if len(methods) == 1:
			desc = {'class': cls, 'module': cls.__module__, 'function_names': methods[0], 'parameters': params, 'series': series}
			
			assert(not n in ta_export_signals.keys())
			
			ta_export_signals.update({n: desc})
		else:
			for m in methods:
				desc = {'class': cls, 'module': cls.__module__, 'function_names': m, 'parameters': params, 'series': series}
				
				assert(not m in ta_export_signals.keys())
				
				ta_export_signals.update({m: desc})
	
	return ta_export_signals
	
if __name__ == "__main__":
	signals = {}
	
	for m in [momentum, trend, volatility, volume, others]:
		signals.update(inspect_(m))

	print_dict_(signals)
