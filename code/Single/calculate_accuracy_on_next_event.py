from __future__ import division
import csv

eventlog = "packages_complete"
pred = 'next'
if pred == 'next':
    csvfile = open('output_files/results/next_activity_and_time_%s.csv' % eventlog, 'r')
elif pred == 'suffix':
    csvfile = open('output_files/results/suffix_and_remaining_time_%s.csv' % eventlog, 'r')
r = csv.reader(csvfile)
next(r, None)  # header
vals = dict()
vals1 = dict()
vals2 = dict()
for row in r:
    if row[3]!='':
        if int(row[1]) > 0:
            l = list()
            l_t = list()
            l_tr = list()
            if row[0] in vals1.keys():
                l = vals.get(row[0])
                l_t = vals1.get(row[0])
                l_tr = vals2.get(row[0])

            if len(row[2]) == 0 and len(row[3]) == 0:
                l.append(1)
            elif len(row[2]) == 0 and len(row[3]) > 0:
                l.append(0)
            elif len(row[2]) > 0 and len(row[3]) == 0:
                l.append(0)
            else:
                if pred == 'next':
                    l.append(int(row[2] == row[3]))
                elif pred == 'suffix':
                    l.append(float(row[5]))
            vals[row[0]] = l
            if pred == 'next':
                l_t.append(float(row[-2]))
                vals1[row[0]] = l_t
                if float(row[9]) > 0:
                    l_tr.append(float(row[-1]))
                vals2[row[0]] = l_tr
            elif pred == 'suffix':
                l_tr.append(float(row[-1]))
                vals2[row[0]] = l_tr

l2 = list()
l3 = list()
l4 = list()
for k in vals2.keys():
    print('{}: {}'.format(k, vals[k]))
    l2.extend(vals[k])
    if pred == 'next':
        l3.extend(vals1[k])
        res1 = sum(vals1[k]) / len(vals1[k])
    else:
        res1 = 'NA'
    if vals2[k]:
        l4.extend(vals2[k])
        res2 = sum(vals2[k]) / len(vals2[k])
    else:
        res2 = 'NA'

    res = sum(vals[k]) / len(vals[k])


    print('{}: {}, {}, {}'.format(k, res, res1, res2))

print('total: {}, t {}, tr {}'.format(sum(l2) / len(l2), sum(l3) / len(l3), sum(l4) / len(l4)))
