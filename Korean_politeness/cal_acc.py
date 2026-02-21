import sys

prob = []
total = 0
corr = 0
for line in open(sys.argv[1]):
    if len(line.strip()) == 0:
        maxp = max(range(len(prob)), key=prob.__getitem__)
        total += 1
        if maxp == 0:
            corr += 1
        prob = []
    else:
        prob.append(float(line.strip().split()[-1]))


print(sys.argv[1])
print(corr/total, corr, total)
