from operator import itemgetter

def tracklet_matching(tracklets, nh, nt, nht, data):
    tintrv = []
    for t_i in nht:
        s, e = tracklets[t_i][0], tracklets[t_i][-1]
        tintrv.append((t_i, int(s[0]), int(e[0])))

    # sort nht by end frame
    tintrv_s = sorted(tintrv,key=itemgetter(2))

    slots = []

    # find overlaps
    for i, i1 in enumerate(tintrv_s):
        cs, ce = i1[1], i1[2]
        ovrlp = [i1]

        for j, i2 in enumerate(tintrv_s):
            if i != j:

                if len(slots) > 0:
                    for s in slots:
                        if i1 in s or i2 in s:
                            continue

                rs, re = i2[1], i2[2]
                if rs <= ce:
                    ovrlp.append(i2)

        slots.append(ovrlp)

    for s in slots:
        print (s)

    print (len(slots))

