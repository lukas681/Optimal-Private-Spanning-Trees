a = input().split()
n = int(a[0]) # players
c = int(a[1]) # cards

# n players, c cards left
def calc(n, c, pos, res):
    if(pos == 0 and c > 0):
        return
    if(c == 0):
        if (pos == 0):
            res.append(True)
            return
        else:
            res.append(False)
            return
    if c < 0: return # should not happen
    # Still cards left
    for i in range(1,  min(c+1, n)): ## playing these cards
        calc(n, c-i, (pos+i+1)%n, res)

results = []
calc(n,c, 1, results)

first = None
toprint = False
for i in results:
    if first == None:
        first = i
        toprint = True
    elif first != i:
        print("maybe")
        toprint = False
        break
if toprint:
    print("yes" if first else "no")