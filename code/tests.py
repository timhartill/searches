
from sortedcontainers import SortedDict
from sortedcontainers import SortedKeyList
import time
import timeit
import random
import heapq

def make_skl(num=100000):
    skl = SortedKeyList(key=lambda x: x[-1])
    for i in range(num):
        skl.add([i, i * 2, i * -1])
    return skl

def make_sd():
    sd = SortedDict()
    for i in range(100000):
        sd[i] = SortedDict()
        for j in range(10):
            sd[i][j] = make_skl(num=5)
    return sd


def make_gdict():
    gdict = SortedDict()
    for i in range(100000):
        gdict[i] = i
    return gdict

skl = make_skl()

# sortedKeyList timings
#timeit.timeit("make_skl()", number=100, globals=globals())  # 13.348112722000224 secs to insert 10 million items

# find by index 
timeit.timeit("skl[0]", number=10000, globals=globals())  # 0.003501657000015257
timeit.timeit("skl[-1]", number=10000, globals=globals())  # 0.0022516469998663524
timeit.timeit("skl[50000]", number=10000, globals=globals())  # 0.013878981000289059
timeit.timeit("skl[99000]", number=10000, globals=globals())  # 0.012645270000120945

# find by index by value
skl.index( [50000, 100000, -50000])  # 49999
timeit.timeit("skl.index([50000, 100000, -50000])", number=10000, globals=globals())  # 0.033184089999849675

# find index by key
skl.bisect_key_left(-50000)  # 49999
timeit.timeit("skl.bisect_key_left(-50000)", number=10000, globals=globals())  # 0.025477457999841135

# pop item by index
skl = make_skl()
timeit.timeit("skl.pop(40000)", number=10000, globals=globals())  # 0.024272700999972585
skl = make_skl()
timeit.timeit("skl.pop(0)", number=10000, globals=globals())  # 0.006296536999798263
skl = make_skl()
timeit.timeit("skl.pop(-1)", number=10000, globals=globals())  # 0.0053877420000389975
skl = make_skl()

timeit.timeit("len(skl)", number=10000, globals=globals())  # 0.0011001340008078841


# SortedDict timings
sd = make_sd()  # ~18 secs for 5 million items

# key exists check
timeit.timeit("0 in sd", number=10000, globals=globals())  # 0.0004508909996729926
timeit.timeit("-1 in sd", number=10000, globals=globals())  # 0.0005510689998118323
timeit.timeit("50000 in sd", number=10000, globals=globals())  # 0.0009081150001293281
timeit.timeit("99999 in sd", number=10000, globals=globals())  # 0.0006532690003950847

timeit.timeit("sd", number=10000, globals=globals())  # 0.00012487599997257348

# .keys[0] vs .peekitem(index=0)[0]
timeit.timeit("f = sd.keys()[0]", number=10000, globals=globals())  # 0.006171058999825618  1st time slower - might be building keys index
timeit.timeit("f = sd.peekitem(index=0)[0]", number=10000, globals=globals())  # 0.0032805209993966855  peekitem[0] optimised vs peekitem[99999]
timeit.timeit("f = sd.peekitem(index=99999)[0]", number=10000, globals=globals())  # 0.023121127000194974
timeit.timeit("f = sd.peekitem(index=50000)[0]", number=10000, globals=globals())  # 0.013238759000159916

# .pop(f) vs .popitem(0)  Note: here the index values are same and in same order as keys
f = sd.peekitem(index=0)[0]  # 0
sd = make_sd()  # ~18 secs for 5 million items
timeit.timeit("for i in range(10000): sd.popitem(index=0)", number=1, globals=globals())  # 0.07188990200120315
sd = make_sd()  # ~18 secs for 5 million items
timeit.timeit("for i in range(10000): sd.pop(i)", number=1, globals=globals())  # 0.0727740910006105

sd = make_sd()  # ~18 secs for 5 million items
timeit.timeit("for i in range(50000, 60000): sd.popitem(index=50000)", number=1, globals=globals())  # 0.10381596799925319
sd = make_sd()  # ~18 secs for 5 million items
timeit.timeit("for i in range(50000, 60000): sd.pop(i)", number=1, globals=globals())  # 0.07648261400026968

f, gbucket = sd.popitem(index=0)  # (f, gbuckets) (0, SortedDict({0: SortedKeyList([0, 0, 0], key=<function make_skl.<locals>.<lambda> at 0x7f8c1c2b3d30>)}))


sd = make_sd()  # each bucket 5 entries
skl = make_skl() # 100k entries
timeit.timeit("sd[10000][3] = SortedKeyList(key=lambda x: x[-1]); sd[10000][3] = skl", number=10, globals=globals())  # 0.013943316000222694

sd = make_sd()  # each bucket 0 entries
timeit.timeit("sd[10000][3] = SortedKeyList(key=lambda x: x[-1]); sd[10000][3].update(skl)", number=10, globals=globals())  # 0.19258439000077487

sd = make_sd()  # each bucket 100k entries, add another 100k entries
skl2 = make_skl(num=100000)
timeit.timeit("sd[10000][3] = skl2; sd[10000][3].update(skl)", number=10, globals=globals())  # 0.9227372120003565

sd = make_sd()
sd[10000][3] = skl
timeit.timeit("len(sd[10000][3]) == 0", number=10000, globals=globals())  # 0.003069798000069568
timeit.timeit("len(skl) == 0", number=10000, globals=globals())  # 0.0011517050006659701
timeit.timeit("len(sd) == 0", number=10000, globals=globals())  # 0.00043866799933311995
timeit.timeit("len(sd[10000]) == 0", number=10000, globals=globals())  # 0.0010653819990693592

sd = SortedDict()
sd[0] = SortedDict()
sd[0][0] = SortedKeyList(key=lambda x: x[-1])
sd[1] = SortedDict()
sd[1][0] = SortedKeyList(key=lambda x: x[-1])
sd[1][0].add([1, 2, 3])
sd[2] = SortedDict()

g = sd.peekitem(index=0)[0]
if not sd[g]:
    sd.pop(g)  # remove if no f buckets
    print("1")
f = sd[g].peekitem(index=0)[0]  # Get the lowest f value
if not sd[g][f]:
    sd[g].pop(f)  # remove if no entries in f bucket ie ready[g][f] 
    print("2")



gdict = make_gdict()
timeit.timeit("for i in range(100000): del gdict[i]", number=1, globals=globals())  #0.08711435802979395
gdict = make_gdict()
timeit.timeit("for i in range(100000): gdict.pop(i)", number=1, globals=globals())  #0.08343088696710765 essentially identical



import util
test_list_full_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
test_list_some_values = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12]
test_list_zeros = [0] * 16
test_list_max = [15] * 16
testlist=[test_list_full_range,test_list_some_values,test_list_zeros,test_list_max]
bits_per_number, num_bytes = calc_bits_bytes(test_list_full_range)

#1.0863950389903039 vs 0.09534033399540931:
timeit.timeit("for t in testlist: encode_numbers_bytes(t, bits_per_number, num_bytes)", number=100000, globals=globals())
timeit.timeit("for t in testlist: bytes(t)", number=100000, globals=globals())

def encdec(t, bits_per_number, num_bytes):
    return 1+1
    bstr = encode_numbers_bytes(t, bits_per_number, num_bytes)
    tout = decode_numbers_bytes(bstr, 16, bits_per_number)

# 0.0267
timeit.timeit("for t in testlist: encdec(t, bits_per_number, num_bytes)", number=100000, globals=globals())


def ifthen(action):
    if action == 'NBS':
        fwd, bwd = True, True
    elif action == 'F':
        fwd, bwd = True, False
    elif action == 'B':
        fwd, bwd = False, True
    elif action == 'A': 
        if action == 'F':
            fwd, bwd = False, True
        else:
            fwd, bwd = True, False
    elif action == 'P':
        if False:
            fwd, bwd = False, True
        else:
            fwd, bwd = True, False
    elif action == 'R':
        if random.choice(['F','B']) == 'F':
            fwd, bwd = True, False
        else:
            fwd, bwd = False, True
    elif action == 'G':
        if True:
            fwd, bwd = False, True
        else:
            fwd, bwd = True, False
    elif action == 'S':
        if False:
            fwd, bwd = False, True
        else:
            fwd, bwd = True, False
    return fwd, bwd

def matchcase(action):
    match action:
        case 'NBS':
            fwd, bwd = True, True
        case'F':
            fwd, bwd = True, False
        case'B':
            fwd, bwd = False, True
        case'A': 
            if action == 'F':
                fwd, bwd = False, True
            else:
                fwd, bwd = True, False
        case'P':
            if False:
                fwd, bwd = False, True
            else:
                fwd, bwd = True, False
        case'R':
            if random.choice(['F','B']) == 'F':
                fwd, bwd = True, False
            else:
                fwd, bwd = False, True
        case'G':
            if True:
                fwd, bwd = False, True
            else:
                fwd, bwd = True, False
        case'S':
            if False:
                fwd, bwd = False, True
            else:
                fwd, bwd = True, False
    return fwd, bwd


timeit.timeit("for a in ['S','G','R','P','A','B','F','NBS']: ifthen(a)", number=1000000, globals=globals()) # 1.386614881004789
timeit.timeit("for a in ['S','G','R','P','A','B','F','NBS']: matchcase(a)", number=1000000, globals=globals()) # 1.563161359008518


tlist = [[i, i*-1] for i in range(1000000)]
random.seed(42)
random.shuffle(tlist)
timeit.timeit("t2 = sorted(tlist)", number=10, globals=globals()) # 16.972086368001328

tlist = [[i, i*-1] for i in range(1000000)]
random.seed(42)
random.shuffle(tlist)
timeit.timeit("random.shuffle(tlist); tlist.sort()", number=10, globals=globals())  # 20.599905465998745

tlist = [[i, i*-1] for i in range(1000000)]
random.seed(42)
random.shuffle(tlist)
timeit.timeit("random.shuffle(tlist); heapq.heapify(tlist)", number=10, globals=globals()) #7.127369664998696

random.seed(42)
tlist = [[random.randint(0,1000000), random.randint(0,1000000)] for i in range(1000000)]
skl = SortedKeyList(key = lambda x: x[-1])
skl.update(tlist)
skl2 = SortedKeyList(key = lambda x: x[0])
timeit.timeit("skl2.clear(); skl2.update(skl)", number=10, globals=globals()) #11.268729454001004

random.seed(42)
tlist = [[random.randint(0,1000000), random.randint(0,1000000)] for i in range(1000000)]
skl = SortedKeyList(key = lambda x: x[-1])
skl.update(tlist)
timeit.timeit("t2 = sorted(tlist)", number=10, globals=globals()) # 16.85926130000007

random.seed(42)
tlist = [[random.randint(0,1000000), random.randint(0,1000000)] for i in range(1000000)]
skl = SortedKeyList(key = lambda x: x[-1])
skl.update(tlist)
timeit.timeit("t2 = list(skl); heapq.heapify(t2)", number=10, globals=globals()) #2.978387428000133


import rust_utils
import timeit
state = [15, 14, 0, 4, 11, 1, 6, 13, 7, 5, 8, 9, 3, 2, 10, 12]
s = SlidingTileProblem(initial_state=state)
_goal_positions = s._goal_positions
state_bytes = bytes(state)
timeit.timeit("s.heuristic(state_bytes, backward=False)", number=100000, globals=globals()) #0.6074775469605811
# maturin develop:
timeit.timeit("rust_utils.heuristic_manhattan(state_bytes, _goal_positions, max_cols=4, degradation=0, make_heuristic_inadmissable=False)", number=100000, globals=globals()) #1.4365476240054704
# maturin develop --release:
timeit.timeit("rust_utils.heuristic_manhattan(state_bytes, _goal_positions, max_cols=4, degradation=0, make_heuristic_inadmissable=False)", number=100000, globals=globals()) #0.13909521396271884
timeit.timeit("heuristic_manhattan(state_bytes, _goal_positions, max_cols=4, degradation=0, make_heuristic_inadmissable=False)", number=100000, globals=globals()) #0.44085584103595465

