"""
A simple script to plot Loss and Error graphs.

Example command: 
python generate_graphs.py --input_csv=results/baseline_pytorch_logistic_regression_train_dev_error_per_epoch.csv --graph_type=error
python generate_graphs.py --input_csv=results/baseline_pytorch_logistic_regression_train_loss_per_minibatch.csv --graph_type=loss

"""

import csv
import argparse
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
import plotly.offline as offline
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', default='input.csv', help="File the csv results")
parser.add_argument('--graph_type', default='loss', help="Use 'error' to plot the train and dev error over epochs, or use 'loss' to plot train loss over iterations")

                    

def plot_loss(input_csv):
	df = pd.read_csv(input_csv)
	print(df)
	trace1 = go.Scatter(
	                    x=df['mini_batch_iteration'], y=df['loss'], # Data
	                    mode='lines', name='loss' # Additional options
	                   )
	layout = go.Layout(title='Loss vs. Num Iterations',plot_bgcolor='rgb(230, 230,230)',xaxis=dict(title='Num Iterations',),yaxis=dict(title='Loss',))
	fig = go.Figure(data=[trace1], layout=layout)

	# Plot data to imagfe
	offline.plot(fig, filename=input_csv, image='png', image_filename=input_csv, output_type='file')

def plot_error(input_csv):
	df = pd.read_csv(input_csv)
	df.loc[:, 'train_accuracy'] = 1 - df
	df.loc[:, 'dev_accuracy'] = 1 - df
	df.rename(columns={'train_accuracy': 'train_error', 'dev_accuracy' : 'dev_error'}, inplace=True)
	print(df)
	trace1 = go.Scatter(
	                    x=df['epoch'], y=df['train_error'], # Data
	                    mode='lines', name='train_error' # Additional options
	                   )
	trace2 = go.Scatter(
                    x=df['epoch'], y=df['dev_error'], # Data
                    mode='lines', name='dev_error' # Additional options
                   )
	layout = go.Layout(title='Train/Dev Error vs. Num Epocs',plot_bgcolor='rgb(230, 230,230)',xaxis=dict(title='Num Iterations',),yaxis=dict(title='Error',))
	fig = go.Figure(data=[trace1, trace2], layout=layout)

	# Plot data in the notebook
	offline.plot(fig, filename=input_csv, image='png', image_filename=input_csv, output_type='file')


if __name__ == '__main__':
	args = parser.parse_args()
	input_csv = args.input_csv
	graph_type = args.graph_type
	print input_csv
	if(graph_type == 'loss'):
		plot_loss(input_csv)
	if(graph_type == 'error'):
		plot_error(input_csv)
	else:
		print("ERROR: Input was an incorrect graph type")





