import sys
import itertools

num = 0
my_dict = {}
for line  in open(sys.argv[1]):
    num += 1
    line = line.strip().split()
    if num == 1:
        temp1 = line
    elif num == 2:
        temp2 = line
    elif num == 3:
        temp3 = line
    else:
        if len(line) > 0:
            if not line[0] in my_dict:
                my_dict[line[0]] = []
            my_dict[line[0]].append(' '.join(line[1:]))

#print(my_dict)
pos = []
for m in my_dict:
    pos.append(temp1.index(m))
#    print(m, my_dict[m])

#print(pos)
keys, values = zip(*my_dict.items())
per = [dict(zip(keys, v)) for v in itertools.product(*values)]
#print(len(per), per)
out = open(sys.argv[2], 'w')
for p in per:
    mid = []
    for a in temp1:
        if a  in p.keys():
            mid.append(p[a])
        else:
            mid.append(a)
    out.write(' '.join(mid)+'\n')
    mid = []
    for a in temp2:
        if a  in p.keys():
            mid.append(p[a])
        else:
            mid.append(a)
    out.write(' '.join(mid)+'\n')
    mid = []
    for a in temp3:
        if a  in p.keys():
            mid.append(p[a])
        else:
            mid.append(a)
    out.write(' '.join(mid)+'\n')
    out.write('\n')

