# This code is modified version of LSTM POS Tagging tutorial by Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

training_data=[]
evaluation_data=[]


def split_to_char(word):
    return list(word)


with open("data/train.txt", "r") as reader:
    for line in reader:
        num_data = line.split("\t")[1].replace('\n', '')
        eng_data = line.split("\t")[0]
        instance = (split_to_char(num_data), split_to_char(eng_data))
        training_data.append(instance)

with open("data/eval.txt", "r") as reader:
    for line in reader:
        num_data = line.split("\t")[1].replace('\n', '')
        eng_data = line.split("\t")[0]
        instance = (split_to_char(num_data), split_to_char(eng_data))
        evaluation_data.append(instance)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


numchar_to_ix = {}
for num, eng in training_data:
    for numchar in num:
        if numchar not in numchar_to_ix:
            numchar_to_ix[numchar] = len(numchar_to_ix)
# print(numchar_to_ix)

engchar_to_ix = {}
for num, eng in training_data:
    for engchar in eng:
        if engchar not in engchar_to_ix:
            engchar_to_ix[engchar] = len(engchar_to_ix)
# print(engchar_to_ix)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Parameters
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
n_epochs = 100
learning_rate = 0.1
model_name = 'bilstm'+"-"+str(n_epochs)+"ep-"+str(EMBEDDING_DIM)+"emb-"+str(HIDDEN_DIM)+"hid"+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = 'models/'+model_name

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        # hidden_dim * 2 for bidirectional lstm
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(numchar_to_ix), len(engchar_to_ix))
model.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# function to return key for any value
def get_key(val):
    for key, value in engchar_to_ix.items():
        if val == value:
            return key

    return "Ω"


def tag_scores_to_text(tag_scores):
    dict_pos = torch.argmax(tag_scores, dim=1)
    text = ""
    for i in dict_pos:
        text+=get_key(i)

    return text

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], numchar_to_ix).to(device)
#     tag_scores = model(inputs)
#     print(tag_scores_to_text(tag_scores))


for epoch in range(1, n_epochs + 1):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, numchar_to_ix).to(device)
        targets = prepare_sequence(tags, engchar_to_ix).to(device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# Save model
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_path)

# Evaluation
with torch.no_grad():
    eval_predictions = []
    for i in range(len(evaluation_data)):
        inputs = prepare_sequence(evaluation_data[i][0], numchar_to_ix).to(device)
        tag_scores = model(inputs)
        eval_predictions.append(tag_scores_to_text(tag_scores))

    with open("predictions/predictions-"+model_name+".txt", "w") as preds:
        for line in eval_predictions:
            preds.write(line+"\n")


