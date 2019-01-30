"""
Implements a simple logistic regression baseline for the task of fake image identification.
"""

import time
import argparse
import os
from data_reader import dataReader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.")
parser.add_argument('--data_folder', default="data/processed_casia2", help="Data folder with preprocessed CASIA data into train/dev/test splits.")
parser.add_argument('--results_folder', default='results/', help="Where to write any results.")
parser.add_argument('--experiment_name', default=None, help="Name for the experiment. Useful for tagging files.")

FLAGS = parser.parse_args()

class_file_name = 'class_names_baselinev2_lr'
confusion_file_name = 'confusion_matrix_baselinev2_lr'


def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("baseline_logistic_regression_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)

def main():
    print("Using Data folder = ", FLAGS.data_folder)

    reader = dataReader(folder=FLAGS.data_folder)

    X_train, Y_train = reader.getTrain()
    X_dev, Y_dev = reader.getDev()
    start_time_secs = time.time()
    print("Starting Logistic Regression training ...", X_train.shape, Y_train.shape)
    classifier = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             multi_class='auto',
                             verbose=1,
                             n_jobs=-1,
                             max_iter=FLAGS.max_iter).fit(X_train, Y_train)
    print("Training done.")
    end_time_secs = time.time()
    training_duration_secs = end_time_secs - start_time_secs
    Y_dev_prediction = classifier.predict(X_dev)

    accuracy = classifier.score(X_dev, Y_dev)

    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nPrediction: {}".format(Y_dev_prediction)
    experiment_result_string += "\nActual Label: {}".format(Y_dev)
    experiment_result_string += "\nAcurracy: {}".format(accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format(
        1.0 * training_duration_secs / FLAGS.max_iter)

    class_names = ['Real', 'Fake']
    classification_report_string = classification_report(Y_dev, Y_dev_prediction, target_names=class_names)
    experiment_result_string += "\nClassification report: {}".format(classification_report_string)

    print(experiment_result_string)

    # Save report to file
    write_contents_to_file(get_experiment_report_filename(), experiment_result_string)






if __name__ == '__main__':
    main()