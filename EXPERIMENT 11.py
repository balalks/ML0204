def consistent(hypo, sample):
    for x, y in zip(hypo, sample):
        if x != '?' and x != y:
            return False
    return True

def candidate_elimination(data):
    G = [['?' for _ in range(len(data[0][0]))]]
    S = data[0][0][:]

    for x, y in data:
        if y == 'Yes':
            for i in range(len(S)):
                if S[i] != x[i]:
                    S[i] = '?'
            G = [g for g in G if consistent(g, x)]
        else:
            G += [[s if s == x[i] else '?' for i, s in enumerate(S)] for i in range(len(S)) if S[i] != '?']

    return S, G

data = [
    (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes'),
]

s_final, g_final = candidate_elimination(data)
print("Final Specific Hypothesis:", s_final)
print("Final General Hypothesis:", g_final)
