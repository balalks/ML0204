def find_s(examples):
    hypothesis = ['0'] * len(examples[0][0])
    for x, label in examples:
        if label == 'Yes':
            for i in range(len(x)):
                if hypothesis[i] == '0':
                    hypothesis[i] = x[i]
                elif hypothesis[i] != x[i]:
                    hypothesis[i] = '?'
    return hypothesis

data = [
    (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes'),
]

print("Final Hypothesis:", find_s(data))
