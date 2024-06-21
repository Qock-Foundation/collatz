import sys
import random
import tqdm

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} N L_min L_max')
        exit(1)
    N = int(sys.argv[1])
    L_min = int(sys.argv[2])
    L_max = int(sys.argv[3])
    subs = []
    for line in sys.stdin.readlines():
        s1, s2 = line.strip().split(' -> ')
        subs.append((s1, s2))
    for _ in tqdm.tqdm(range(N)):
        L = random.randint(L_min, L_max)
        chars = 'abefg'
        out1 = 'c' + ''.join(random.choice(chars) for _ in range(L - 2)) + 'd'
        occ = []
        for s1, s2 in subs:
            for i in range(L - len(s1) + 1):
                if out1[i:i + len(s1)] == s1:
                    occ.append((i, s1, s2))
        ind, s1, s2 = random.choice(occ)
        out2 = out1[:ind] + s2 + out1[ind + len(s1):]
        print(out1, out2)
    print(N)
