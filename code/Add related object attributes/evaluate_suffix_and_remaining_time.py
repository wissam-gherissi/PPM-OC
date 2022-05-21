from __future__ import division

from operator import itemgetter

from keras.models import load_model
import csv
import copy
import numpy as np
import distance
# from itertools import izip
from jellyfish._jellyfish import damerau_levenshtein_distance
import unicodecsv
from numpy.compat import unicode
from six import unichr
from sklearn import metrics
from math import sqrt
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

eventlog = "orders_el_new.csv"
secondary_el = "items_el.csv"

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers

csvfile_2 = open('../data/%s' % secondary_el, 'r')
spamreader_2 = csv.reader(csvfile_2, delimiter=',', quotechar='|')
next(spamreader_2, None)

obj_rows = []
for row in spamreader_2:
    obj_rows.append(row)
csvfile_2.close()

ascii_offset = 64

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
times = []
times2 = []
times3 = []
numlines = 0
casestarttime = None
lasteventtime = None
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0] != lastcase:
        caseids.append(row[0])
        final_time = datetime.fromtimestamp(time.mktime(t))
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
        line = ''
        times = []
        times2 = []
        times3 = []
        numlines += 1

    related_objs = row[3].split(':')
    for obj in related_objs:
        related_rows = [row_item for row_item in obj_rows if row_item[0] == obj]
        rows_time_sorted = sorted(related_rows, key=itemgetter(2))
        idx = len(rows_time_sorted) - 1
        t_obj = datetime.min
        while idx >= 0:
            dt = time.strptime(rows_time_sorted[idx][2], "%Y-%m-%d %H:%M:%S")
            dt = datetime.fromtimestamp(time.mktime(dt))
            if dt < datetime.fromtimestamp(time.mktime(t)):
                t_obj = dt
                break
            idx -= 1
        if final_time <= t_obj:
            final_time = t_obj

    timediff3 = datetime.fromtimestamp(time.mktime(t)) - final_time
    times3.append(timediff3.total_seconds())

    line += unichr(int(row[1]) + ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
numlines += 1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x) - 1] - y, x))), timeseqs2)))
print('divisor3: {}'.format(divisor3))
divisor4 = np.mean([item for sublist in timeseqs3 for item in sublist])
print('divisor4: {}'.format(divisor4))

elems_per_fold = int(round(numlines / 3))
fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]

fold2 = lines[elems_per_fold:2 * elems_per_fold]
fold2_c = caseids[elems_per_fold:2 * elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2 * elems_per_fold]

lines = fold1 + fold2
caseids = fold1_c + fold2_c
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x + '!', lines))
maxlen = max(map(lambda x: len(x), lines))

# chars = map(lambda x : set(x),lines)
chars = list(set().union(*map(lambda x: set(x), lines)))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []  # relative time since previous event
timeseqs2 = []  # relative time since case start
timeseqs3 = []  # absolute time of previous event
timeseqs4 = []
times = []
times2 = []
times3 = []
times4 = []
numlines = 0
casestarttime = None
lasteventtime = None
final_time = datetime.min
csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0] != lastcase:
        caseids.append(row[0])
        final_time = datetime.fromtimestamp(time.mktime(t))
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            timeseqs4.append(times4)

        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []
        numlines += 1

    related_objs = row[3].split(':')
    for obj in related_objs:
        related_rows = [row_item for row_item in obj_rows if row_item[0] == obj]
        rows_time_sorted = sorted(related_rows, key=itemgetter(2))
        idx = len(rows_time_sorted) - 1
        t_obj = datetime.min
        while idx >= 0:
            dt = time.strptime(rows_time_sorted[idx][2], "%Y-%m-%d %H:%M:%S")
            dt = datetime.fromtimestamp(time.mktime(dt))
            if dt < datetime.fromtimestamp(time.mktime(t)):
                t_obj = dt
                break
            idx -= 1
        if final_time < t_obj:
            final_time = t_obj

    timediff4 = datetime.fromtimestamp(time.mktime(t)) - final_time
    times4.append(timediff4.total_seconds())

    line += unichr(int(row[1]) + ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)

numlines += 1

fold3 = lines[2 * elems_per_fold:]
fold3_c = caseids[2 * elems_per_fold:]
fold3_t = timeseqs[2 * elems_per_fold:]
fold3_t2 = timeseqs2[2 * elems_per_fold:]
fold3_t3 = timeseqs3[2 * elems_per_fold:]
fold3_t4 = timeseqs4[2 * elems_per_fold:]

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
lines_t4 = fold3_t4

# set parameters
predict_size = maxlen

# load model, set this to the model generated by train.py
model = load_model('output_files/models/model_332-2.12.h5')


# define helper functions
def encode(sentence, times, times3,times4, maxlen=maxlen):
    num_features = len(chars) + 6
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t] - midnight
        multiset_abstraction = Counter(sentence[:t + 1])
        for c in chars:
            if c == char:
                X[0, t + leftpad, char_indices[c]] = 1
        X[0, t + leftpad, len(chars)] = t + 1
        X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
        X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
        X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
        X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        X[0, t + leftpad, len(chars) + 5] = times4[t] / divisor4
    return X


def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0
    for prediction in predictions:
        if (prediction >= maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol


one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

# make predictions
with open('output_files/results/suffix_and_remaining_time_%s' % eventlog, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
                         "Ground truth times", "Predicted times", "RMSE", "MAE"])
    for prefix_size in range(2, maxlen):
        print(prefix_size)
        for line, caseid, times, times2, times3, times4 in zip(lines, caseids, lines_t, lines_t2, lines_t3, lines_t4):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            cropped_times4 = times4[:prefix_size]
            if len(times2) < prefix_size:
                continue  # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = times2[prefix_size - 1]
            case_end_time = times2[len(times2) - 1]
            ground_truth_t = case_end_time - ground_truth_t
            predicted = ''
            total_predicted_time = 0
            for i in range(predict_size):
                enc = encode(cropped_line, cropped_times, cropped_times3, cropped_times4)
                y = model.predict(enc, verbose=0)  # make predictions
                # split predictions into seperate activity and time predictions
                y_char = y[0][0]
                y_t = y[1][0][0]
                prediction = getSymbol(y_char)  # undo one-hot encoding
                cropped_line += prediction
                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)
                if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                    one_ahead_pred.append(total_predicted_time)
                    one_ahead_gt.append(ground_truth_t)
                    print('! predicted, end case')
                    break
                y_t = y_t * divisor3
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                y_t_2 = y_t * divisor4
                cropped_times4.append(cropped_times4[-1] + timedelta(seconds=y_t_2).total_seconds())
                total_predicted_time = total_predicted_time + y_t
                predicted += prediction
            output = []
            if len(ground_truth) > 0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(unicode(ground_truth))
                output.append(unicode(predicted))
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                dls = 1 - (damerau_levenshtein_distance(unicode(predicted), unicode(ground_truth)) / max(len(predicted),
                                                                                                         len(ground_truth)))
                if dls < 0:
                    dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append(ground_truth_t)
                output.append(total_predicted_time)
                output.append('')
                output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                # output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                spamwriter.writerow(output)
