"""
NAME: data.py

PURPOSE: data class: noisy observations on locations
"""


class Data():

	# Assume additive Gaussian observation error -> std_dev
	def __init__(self, locations, observations, variance = 0.):
		self.locations = locations
		self.observations = observations
		self.variance = variance