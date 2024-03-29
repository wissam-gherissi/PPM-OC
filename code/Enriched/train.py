from __future__ import print_function, division

from operator import itemgetter
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from tensorflow.python.keras.layers import
from collections import Counter
import numpy as np
import random
import sys
import copy
import csv
import time
# from itertools import izip
from datetime import datetime

from numpy.compat import unicode
from six import unichr

eventlog = "orders_complete"

np.random.seed(42)

########################################################################################
#
# this part of the code opens the file, reads it into three following variables
#

lines = []  # these are all the activity seq
Ptimeseqs = []  # time sequences (differences between two events)
Ptimeseqs2 = []  # time sequences (differences between the current and first)
nb_itemseqs = []

# helper variables
lastcase = ''
line = ''
firstLine = True
Ptimes = []
Ptimes2 = []
nb_items = []

numlines = 0
Pcasestarttime = None
Plasteventtime = None

csvfile = open('../../data/%s.csv' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers

ascii_offset = 64

for row in spamreader:  # the rows are "CaseID,ActivityID,CompleteTimestamp,related_objects"
    #    t = row[2]
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # creates a datetime object from row[2]
    t = datetime.fromtimestamp(time.mktime(t))
    nb = len(row[3].split(':'))

    if row[0] != lastcase:  # 'lastcase' is to save the last executed case for the loop
        Pcasestarttime = t
        Plasteventtime = t

        lastcase = row[0]

        if not firstLine:
            lines.append(line)
            Ptimeseqs.append(Ptimes)
            Ptimeseqs2.append(Ptimes2)
            nb_itemseqs.append(nb_items)

        line = ''
        Ptimes = []
        Ptimes2 = []
        nb_items = []

        numlines += 1

    # line = line + ' ' + row[1]
    line += unichr(int(row[1]) + ascii_offset)
    Ptimesincelastevent = t - Plasteventtime
    Ptimesincecasestart = t - Pcasestarttime
    Ptimediff = 86400 * Ptimesincelastevent.days + Ptimesincelastevent.seconds
    Ptimediff2 = 86400 * Ptimesincecasestart.days + Ptimesincecasestart.seconds
    Ptimes.append(Ptimediff)
    Ptimes2.append(Ptimediff2)
    nb_items.append(nb)
    Plasteventtime = t

    firstLine = False

# add last case
lines.append(line)
Ptimeseqs.append(Ptimes)
Ptimeseqs2.append(Ptimes2)
nb_itemseqs.append(nb_items)
numlines += 1

PtimeseqsF = []
for seq in Ptimeseqs2:
    l = []
    for t in seq:
        l.append(seq[-1] - t)
    PtimeseqsF.append(l)


########################################

divisor = np.mean([item for sublist in Ptimeseqs for item in sublist])  # average time between events
print('divisor: {}'.format(divisor))
divisor2 = np.mean(
    [item for sublist in Ptimeseqs2 for item in sublist])  # average time between current and first events
print('divisor2: {}'.format(divisor2))
divisorTR = np.mean(
    [item for sublist in PtimeseqsF for item in sublist])  # average time instance remaining
print('divisor2: {}'.format(divisorTR))
#########################################################################################################

# separate training data into 3 parts

indices = np.random.permutation(numlines - 1)
elems_per_fold = int(round(numlines / 3))

idx1 = indices[:elems_per_fold]
idx2 = indices[elems_per_fold:2 * elems_per_fold]
idx3 = indices[2 * elems_per_fold:]

fold1 = list(itemgetter(*idx1)(lines))
fold1_t = list(itemgetter(*idx1)(Ptimeseqs))
fold1_t2 = list(itemgetter(*idx1)(Ptimeseqs2))
fold1_t3 = list(itemgetter(*idx1)(nb_itemseqs))
fold1_t4 = list(itemgetter(*idx1)(PtimeseqsF))

fold2 = list(itemgetter(*idx2)(lines))
fold2_t = list(itemgetter(*idx2)(Ptimeseqs))
fold2_t2 = list(itemgetter(*idx2)(Ptimeseqs2))
fold2_t3 = list(itemgetter(*idx2)(nb_itemseqs))
fold2_t4 = list(itemgetter(*idx2)(PtimeseqsF))

fold3 = list(itemgetter(*idx3)(lines))
fold3_t = list(itemgetter(*idx3)(Ptimeseqs))
fold3_t2 = list(itemgetter(*idx3)(Ptimeseqs2))
fold3_t3 = list(itemgetter(*idx3)(nb_itemseqs))
fold3_t4 = list(itemgetter(*idx3)(PtimeseqsF))


# leave away fold3 for now
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4


step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x + '!', lines))  # put delimiter symbol
maxlen = max(map(lambda x: len(x), lines))  # find maximum line size

# next lines here to get all possible characters for events and annotate them with numbers
# chars = map(lambda x: set(x),lines)
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

csvfile = open('../../data/%s.csv' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
lastcase = ''
line = ''
firstLine = True
lines = []
Ptimeseqs = []
Ptimeseqs2 = []
Ptimeseqs3 = []
Ptimeseqs4 = []
nb_itemseqs = []

Ptimes = []
Ptimes2 = []
Ptimes3 = []
Ptimes4 = []
nb_items = []

numlines = 0
Pcasestarttime = None
Plasteventtime = None

for row in spamreader:
    #    t = row[2]
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    t = datetime.fromtimestamp(time.mktime(t))
    nb = len(row[3].split(':'))
    if row[0] != lastcase:
        Stime = t
        Pcasestarttime = t
        Plasteventtime = t
        lastcase = row[0]
        if not firstLine:
            lines.append(line)
            Ptimeseqs.append(Ptimes)
            Ptimeseqs2.append(Ptimes2)
            Ptimeseqs3.append(Ptimes3)
            Ptimeseqs4.append(Ptimes4)
            nb_itemseqs.append(nb_items)

        line = ''
        Ptimes = []
        Ptimes2 = []
        Ptimes3 = []
        Ptimes4 = []
        nb_items = []

        numlines += 1

    # line += row[1]
    line += unichr(int(row[1]) + ascii_offset)
    Ptimesincelastevent = t - Plasteventtime
    Ptimesincecasestart = t - Pcasestarttime
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = t - midnight
    Ptimediff = 86400 * Ptimesincelastevent.days + Ptimesincelastevent.seconds
    Ptimediff2 = 86400 * Ptimesincecasestart.days + Ptimesincecasestart.seconds
    Ptimediff3 = timesincemidnight.seconds  # this leaves only time even occured after midnight
    Ptimediff4 = t.weekday()  # day of the week
    Ptimes.append(Ptimediff)
    Ptimes2.append(Ptimediff2)
    Ptimes3.append(Ptimediff3)
    Ptimes4.append(Ptimediff4)
    nb_items.append(nb)
    Plasteventtime = t
    firstLine = False

# add last case
lines.append(line)
Ptimeseqs.append(Ptimes)
Ptimeseqs2.append(Ptimes2)
Ptimeseqs3.append(Ptimes3)
Ptimeseqs4.append(Ptimes4)
nb_itemseqs.append(nb_items)

PtimeseqsF = []
for seq in Ptimeseqs2:
    l = []
    for t in seq:
        l.append(seq[-1] - t)
    PtimeseqsF.append(l)


numlines += 1

# elems_per_fold = int(round(numlines / 3))
fold1 = list(itemgetter(*idx1)(lines))
fold1_t = list(itemgetter(*idx1)(Ptimeseqs))
fold1_t2 = list(itemgetter(*idx1)(Ptimeseqs2))
fold1_t3 = list(itemgetter(*idx1)(Ptimeseqs3))
fold1_t4 = list(itemgetter(*idx1)(Ptimeseqs4))
fold1_t5 = list(itemgetter(*idx1)(nb_itemseqs))
fold1_t6 = list(itemgetter(*idx1)(PtimeseqsF))

with open('./output_files/folds/fold1.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold1, fold1_t):
        spamwriter.writerow([unicode(s) + '#{}'.format(t) for s, t in zip(row, timeseq)])

fold2 = list(itemgetter(*idx2)(lines))
fold2_t = list(itemgetter(*idx2)(Ptimeseqs))
fold2_t2 = list(itemgetter(*idx2)(Ptimeseqs2))
fold2_t3 = list(itemgetter(*idx2)(Ptimeseqs3))
fold2_t4 = list(itemgetter(*idx2)(Ptimeseqs4))
fold2_t5 = list(itemgetter(*idx2)(nb_itemseqs))
fold2_t6 = list(itemgetter(*idx2)(PtimeseqsF))
with open('output_files/folds/fold2.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold2, fold2_t):
        spamwriter.writerow([unicode(s) + '#{}'.format(t) for s, t in zip(row, timeseq)])

fold3 = list(itemgetter(*idx3)(lines))
fold3_t = list(itemgetter(*idx3)(Ptimeseqs))
fold3_t2 = list(itemgetter(*idx3)(Ptimeseqs2))
fold3_t3 = list(itemgetter(*idx3)(Ptimeseqs3))
fold3_t4 = list(itemgetter(*idx3)(Ptimeseqs4))
fold3_t5 = list(itemgetter(*idx3)(nb_itemseqs))
fold3_t6 = list(itemgetter(*idx3)(PtimeseqsF))
with open('output_files/folds/fold3.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold3, fold3_t):
        spamwriter.writerow(
            [unicode(s).encode("utf-8") + '#{}'.format(t).encode('utf-8') for s, t in zip(row, timeseq)])

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4
lines_t5 = fold1_t5 + fold2_t5
lines_t6 = fold1_t6 + fold2_t6


step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x + '!', lines))

sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []
sentences_t5 = []
sentences_t6 = []


next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []
next_chars_t5 = []
next_chars_t6 = []


for line, line_t, line_t2, line_t3, line_t4, line_t5, line_t6 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4, lines_t5, lines_t6):
    for i in range(0, len(line), step):
        if i == 0:
            continue

        # we add iteratively, first symbol of the line, then two first, three...

        sentences.append(line[0: i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        sentences_t5.append(line_t5[0:i])
        sentences_t6.append(line_t6[0:i])
        next_chars.append(line[i])
        if i == len(line) - 1:  # special case to deal time of end character
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
            next_chars_t5.append(0)
            next_chars_t6.append(0)


        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])
            next_chars_t5.append(line_t5[i])
            next_chars_t6.append(line_t6[i])

print('nb sequences:', len(sentences))

print('Vectorization...')
num_features = len(chars) + 6
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)
y_tr = np.zeros((len(sentences)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    leftpad = maxlen - len(sentence)
    next_t = next_chars_t[i]
    next_tr = next_chars_t5[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    sentence_t5 = sentences_t5[i]

    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t + 1])
        for c in chars:
            if c == char:  # this will encode present events to the right places
                X[i, t + leftpad, char_indices[c]] = 1
        X[i, t + leftpad, len(chars)] = t + 1
        X[i, t + leftpad, len(chars) + 1] = sentence_t[t] / divisor
        X[i, t + leftpad, len(chars) + 2] = sentence_t2[t] / divisor2
        X[i, t + leftpad, len(chars) + 3] = sentence_t3[t] / 86400
        X[i, t + leftpad, len(chars) + 4] = sentence_t4[t] / 7
        X[i, t + leftpad, len(chars) + 5] = sentence_t5[t]

    for c in target_chars:
        if c == next_chars[i]:
            y_a[i, target_char_indices[c]] = 1 - softness
        else:
            y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)
    y_t[i] = next_t / divisor
    y_tr[i] = next_tr / divisorTR
    np.set_printoptions(threshold=sys.maxsize)

# build the model: 
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(
    main_input)  # the shared layer
b1 = tf.keras.layers.BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
    b1)  # the layer specialized in activity prediction
b2_1 = tf.keras.layers.BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
    b1)  # the layer specialized in time prediction
b2_2 = tf.keras.layers.BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(
    b2_1)
l2_3 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
    b1)  # the layer specialized in time remaining prediction
b2_3 = tf.keras.layers.BatchNormalization()(l2_3)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(
    b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

timeR_output = Dense(1, kernel_initializer='glorot_uniform', name='timeR_output')(b2_3)

model = Model(inputs=[main_input], outputs=[act_output, time_output, timeR_output])

model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae', 'timeR_output': 'mae'}, optimizer='nadam')
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
model_checkpoint = ModelCheckpoint('output_files/models/model_'+eventlog+'_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=0, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)

model.fit(X, {'act_output': y_a, 'time_output': y_t, 'timeR_output': y_tr}, validation_split=0.2, verbose=2,
          callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=500)
