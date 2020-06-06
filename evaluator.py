model_name = "predictions-bilstm-100ep-32emb-32hid-2020-06-06_18-53-46"

preds = []
eval = []

with open("predictions/"+model_name+".txt", "r") as reader:
    for line in reader:
        preds.append(line.replace('\n', ''))

with open("data/eval.txt", "r") as reader:
    for line in reader:
        eng_data = line.split("\t")[0]
        eval.append(eng_data)

correct = 0
total = 0

if len(preds)==len(eval):
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if preds[i][j] == eval[i][j]:
                correct+=1
            total+=1

accuracy = correct / total
print("Correct chars: {}".format(correct))
print("Total chars: {}".format(total))
print("Accuracy: {:.4f}".format(accuracy))