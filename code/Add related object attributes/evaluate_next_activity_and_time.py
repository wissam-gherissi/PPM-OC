from __future__ import division

import random
from operator import itemgetter

from keras.models import load_model
import csv
import copy
import numpy as np
import distance
# from itertools import izip
from jellyfish._jellyfish import damerau_levenshtein_distance
from numpy.compat import unicode
from six import unichr
from sklearn import metrics
import time
from datetime import datetime, timedelta
from collections import Counter
from tqdm import tqdm

eventlog = "orders_el_new.csv"
secondary_el = "items_el_new.csv"

np.random.seed(42)

csvfile = open('../../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 64

csvfile_2 = open('../../data/%s' % secondary_el, 'r')
spamreader_2 = csv.reader(csvfile_2, delimiter=',', quotechar='|')
next(spamreader_2, None)

obj_rows = []
for row in spamreader_2:
    t_row = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # creates a datetime object from row[2]
    row[2] = datetime.fromtimestamp(time.mktime(t_row))
    obj_rows.append(row)
csvfile_2.close()

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
Ptimeseqs = []
Ptimeseqs2 = []
Stimeseqs = []
Stimeseqs2 = []
Stimeseqs3 = []

Ptimes = []
Ptimes2 = []
Stimes = []
Stimes2 = []
Stimes3 = []
numlines = 0
Pcasestarttime = None
Plasteventtime = None
for row in tqdm(spamreader):
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    t = datetime.fromtimestamp(time.mktime(t))
    if row[0] != lastcase:
        caseids.append(row[0])
        Stime = t
        Slasteventtime = t
        Scasestarttime = t
        Pcasestarttime = t
        Plasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            Ptimeseqs.append(Ptimes)
            Ptimeseqs2.append(Ptimes2)
            Stimeseqs.append(Stimes)
            Stimeseqs2.append(Stimes2)
            Stimeseqs3.append(Stimes3)
        line = ''
        Ptimes = []
        Ptimes2 = []
        Stimes = []
        Stimes2 = []
        Stimes3 = []
        numlines += 1

    related_objs = row[3].split(':')
    for obj in related_objs:
        related_rows = [row_item for row_item in obj_rows if row_item[0] == obj and t > row_item[2] >= Plasteventtime]
        rows_time_sorted = sorted(related_rows, key=itemgetter(2))
        if len(rows_time_sorted) > 0:
            if Stime < rows_time_sorted[-1][2]:
                Stime = rows_time_sorted[-1][2]

    if Stime != t:
        obj_trace = []
        while not obj_trace:
            obj_id = random.choice(related_objs)
            obj_trace = [row_item for row_item in obj_rows if row_item[0] == obj_id and t > row_item[2]]

        trace_time_sorted = sorted(obj_trace, key=itemgetter(2))
        if len(obj_trace) > 1:
            Slasteventtime = obj_trace[-2][2]
        else:
            Slasteventtime = obj_trace[0][2]
        Scasestarttime = obj_trace[0][2]

    Stimediff = t - Stime
    Stimes.append(Stimediff.total_seconds())
    Stimediff2 = t - Scasestarttime
    Stimes2.append(Stimediff2.total_seconds())
    Stimediff3 = Stime - Slasteventtime
    Stimes3.append(Stimediff3.total_seconds())

    line += unichr(int(row[1]) + ascii_offset)
    Ptimesincelastevent = t - Plasteventtime
    Ptimesincecasestart = t - Pcasestarttime
    Ptimediff = 86400 * Ptimesincelastevent.days + Ptimesincelastevent.seconds
    Ptimediff2 = 86400 * Ptimesincecasestart.days + Ptimesincecasestart.seconds
    Ptimes.append(Ptimediff)
    Ptimes2.append(Ptimediff2)
    Plasteventtime = t
    Stime = t
    Slasteventtime = t
    Scasestarttime = t
    firstLine = False

# add last case
lines.append(line)
Ptimeseqs.append(Ptimes)
Ptimeseqs2.append(Ptimes2)
Stimeseqs.append(Stimes)
Stimeseqs2.append(Stimes2)
Stimeseqs3.append(Stimes3)
numlines += 1

divisor = np.mean([item for sublist in Ptimeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in Ptimeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean([item for sublist in Stimeseqs for item in sublist])
print('divisor3: {}'.format(divisor3))
divisor4 = np.mean([item for sublist in Stimeseqs2 for item in sublist])
print('divisor4: {}'.format(divisor4))
divisor5 = np.mean([item for sublist in Stimeseqs3 for item in sublist])
print('divisor5: {}'.format(divisor5))

indices = np.random.permutation(numlines-1)
elems_per_fold = int(round(numlines / 3))

idx1 = indices[:2 * elems_per_fold]
idx2 = indices[2 * elems_per_fold:4 * elems_per_fold]
idx3 = indices[4 * elems_per_fold:]

fold1 = list(itemgetter(*idx1)(lines))
fold1_c = list(itemgetter(*idx1)(caseids))
fold1_t = list(itemgetter(*idx1)(Ptimeseqs))
fold1_t2 = list(itemgetter(*idx1)(Ptimeseqs2))
fold1_t3 = list(itemgetter(*idx1)(Stimeseqs))
fold1_t4 = list(itemgetter(*idx1)(Stimeseqs2))
fold1_t5 = list(itemgetter(*idx1)(Stimeseqs3))


fold2 = list(itemgetter(*idx2)(lines))
fold2_c = list(itemgetter(*idx2)(caseids))
fold2_t = list(itemgetter(*idx2)(Ptimeseqs))
fold2_t2 = list(itemgetter(*idx2)(Ptimeseqs2))
fold2_t3 = list(itemgetter(*idx2)(Stimeseqs))
fold2_t4 = list(itemgetter(*idx2)(Stimeseqs2))
fold2_t5 = list(itemgetter(*idx2)(Stimeseqs3))

lines = fold1 + fold2
caseids = fold1_c + fold2_c
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4
lines_t5 = fold1_t5 + fold2_t5

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
Ptimeseqs = []  # relative time since previous event
Ptimeseqs2 = []  # relative time since case start
Ptimeseqs3 = []  # absolute time of previous event
Stimeseqs = []
Stimeseqs2 = []
Stimeseqs3 = []

Ptimes = []
Ptimes2 = []
Ptimes3 = []
Stimes = []
Stimes2 = []
Stimes3 = []

numlines = 0
Pcasestarttime = None
Plasteventtime = None
Stime = datetime.min
csvfile = open('../../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
for row in tqdm(spamreader):
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    t = datetime.fromtimestamp(time.mktime(t))
    if row[0] != lastcase:
        caseids.append(row[0])
        Stime = t
        Pcasestarttime = t
        Plasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            Ptimeseqs.append(Ptimes)
            Ptimeseqs2.append(Ptimes2)
            Ptimeseqs3.append(Ptimes3)
            Stimeseqs.append(Stimes)
            Stimeseqs2.append(Stimes2)
            Stimeseqs3.append(Stimes3)
        line = ''
        Ptimes = []
        Ptimes2 = []
        Ptimes3 = []
        Stimes = []
        Stimes2 = []
        Stimes3 = []
        numlines += 1

    related_objs = row[3].split(':')
    for obj in related_objs:
        related_rows = [row_item for row_item in obj_rows if row_item[0] == obj and t > row_item[2] >= Plasteventtime]
        rows_time_sorted = sorted(related_rows, key=itemgetter(2))
        if len(rows_time_sorted) > 0:
            if Stime < rows_time_sorted[-1][2]:
                Stime = rows_time_sorted[-1][2]

    if Stime != t:
        obj_trace = []
        while not obj_trace:
            obj_id = random.choice(related_objs)
            obj_trace = [row_item for row_item in obj_rows if row_item[0] == obj_id and t > row_item[2]]

        trace_time_sorted = sorted(obj_trace, key=itemgetter(2))
        if len(obj_trace) > 1:
            Slasteventtime = obj_trace[-2][2]
        else:
            Slasteventtime = obj_trace[0][2]
        Scasestarttime = obj_trace[0][2]

    Stimediff = t - Stime
    Stimes.append(Stimediff.total_seconds())
    Stimediff2 = t - Scasestarttime
    Stimes2.append(Stimediff2.total_seconds())
    Stimediff3 = Stime - Slasteventtime
    Stimes3.append(Stimediff3.total_seconds())

    line += unichr(int(row[1]) + ascii_offset)
    Ptimesincelastevent = t - Plasteventtime
    Ptimesincecasestart = t - Pcasestarttime
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = t - midnight
    Ptimediff = 86400 * Ptimesincelastevent.days + Ptimesincelastevent.seconds
    Ptimediff2 = 86400 * Ptimesincecasestart.days + Ptimesincecasestart.seconds
    Ptimes.append(Ptimediff)
    Ptimes2.append(Ptimediff2)
    Ptimes3.append(t)
    Plasteventtime = t
    firstLine = False

# add last case
lines.append(line)
Ptimeseqs.append(Ptimes)
Ptimeseqs2.append(Ptimes2)
Ptimeseqs3.append(Ptimes3)

Stimeseqs.append(Stimes)
Stimeseqs2.append(Stimes2)
Stimeseqs3.append(Stimes3)
numlines += 1

fold3 = list(itemgetter(*idx3)(lines))
fold3_c = list(itemgetter(*idx3)(caseids))
fold3_t = list(itemgetter(*idx3)(Ptimeseqs))
fold3_t2 = list(itemgetter(*idx3)(Ptimeseqs2))
fold3_t3 = list(itemgetter(*idx3)(Ptimeseqs3))
fold3_t4 = list(itemgetter(*idx3)(Stimeseqs))
fold3_t5 = list(itemgetter(*idx3)(Stimeseqs2))
fold3_t6 = list(itemgetter(*idx3)(Stimeseqs3))

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
lines_t4 = fold3_t4
lines_t5 = fold3_t5
lines_t6 = fold3_t6

# set parameters
predict_size = 1

# load model, set this to the model generated by train.py
model = load_model('output_files/models/model_47-1.79.h5')


# define helper functions
def encode(sentence, times, times3, times4, times5, times6, maxlen=maxlen):
    num_features = len(chars) + 8
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
        X[0, t + leftpad, len(chars) + 5] = times4[t] / divisor3
        X[0, t + leftpad, len(chars) + 6] = times5[t] / divisor4
        X[0, t + leftpad, len(chars) + 7] = times6[t] / divisor5

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
with open('output_files/results/next_activity_and_time_%s' % eventlog, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
                         "Ground truth times", "Predicted times", "RMSE", "MAE"])
    for prefix_size in range(2, maxlen):
        print(prefix_size)
        for line, caseid, Ptimes, Ptimes3, Stimes, Stimes2, Stimes3 in zip(lines, caseids, lines_t, lines_t3, lines_t4, lines_t5, lines_t6):
            Ptimes.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = Ptimes[:prefix_size]
            cropped_times3 = Ptimes3[:prefix_size]
            cropped_times4 = Stimes[:prefix_size]
            cropped_times5 = Stimes2[:prefix_size]
            cropped_times6 = Stimes3[:prefix_size]

            if '!' in cropped_line:
                continue  # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = Ptimes[prefix_size:prefix_size + predict_size]
            predicted = ''
            predicted_t = []
            for i in range(predict_size):
                if len(ground_truth) <= i:
                    continue
                enc = encode(cropped_line, cropped_times, cropped_times3, cropped_times4, cropped_times5, cropped_times6)
                y = model.predict(enc, verbose=0)
                y_char = y[0][0]
                y_t = y[1][0][0]
                prediction = getSymbol(y_char)
                cropped_line += prediction
                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)
                y_t1 = y_t * divisor
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t1))
                y_t_2 = y_t * divisor3
                cropped_times4.append(cropped_times4[-1] + timedelta(seconds=y_t_2).total_seconds())
                y_t_3 = y_t * divisor4
                cropped_times5.append(cropped_times5[-1] + timedelta(seconds=y_t_3).total_seconds())
                y_t_4 = y_t * divisor5
                cropped_times6.append(cropped_times6[-1] + timedelta(seconds=y_t_4).total_seconds())
                predicted_t.append(y_t1)
                if i == 0:
                    if len(ground_truth_t) > 0:
                        one_ahead_pred.append(y_t1)
                        one_ahead_gt.append(ground_truth_t[0])
                if i == 1:
                    if len(ground_truth_t) > 1:
                        two_ahead_pred.append(y_t1)
                        two_ahead_gt.append(ground_truth_t[1])
                if i == 2:
                    if len(ground_truth_t) > 2:
                        three_ahead_pred.append(y_t1)
                        three_ahead_gt.append(ground_truth_t[2])
                if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                    print('! predicted, end case')
                    break
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
                output.append('; '.join(str(x) for x in ground_truth_t))
                output.append('; '.join(str(x) for x in predicted_t))
                if len(predicted_t) > len(
                        ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                    predicted_t = predicted_t[:len(ground_truth_t)]
                if len(ground_truth_t) > len(
                        predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                    predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                    output.append('')
                    output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                    # output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                else:
                    output.append('')
                    output.append('')
                    output.append('')
                spamwriter.writerow(output)
