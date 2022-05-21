from __future__ import division
import csv

eventlog = "orders_el_new.csv"
csvfile = open('output_files/results/next_activity_and_time_%s' % eventlog, 'r')
r = csv.reader(csvfile)
next(r, None)  # header
vals = dict()
vals1 = dict()
for row in r:
    if int(row[1]) > 0:
        l = list()
        l1 = list()
        if row[0] in vals1.keys():
            l = vals.get(row[0])
            l1 = vals1.get(row[0])
        if len(row[2]) == 0 and len(row[3]) == 0:
            l.append(1)
        elif len(row[2]) == 0 and len(row[3]) > 0:
            l.append(0)
        elif len(row[2]) > 0 and len(row[3]) == 0:
            l.append(0)
        else:
            l.append(int(row[2] == row[3]))
        vals[row[0]] = l
        l1.append(float(row[-1]))
        vals1[row[0]] = l1

l2 = list()
l3 = list()
for k in vals1.keys():
    print('{}: {}'.format(k, vals[k]))
    l2.extend(vals[k])
    l3.extend(vals1[k])
    res1 = sum(vals1[k]) / len(vals1[k])
    res = sum(vals[k]) / len(vals[k])

    print('{}: {},{}'.format(k, res, res1))

print('total: {},{}'.format(sum(l2) / len(l2), sum(l3) / len(l3)))
