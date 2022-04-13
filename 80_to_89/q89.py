import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

#実際の言語モデル(BERT)と、単語をidに変換するtokenizerを用意
bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

#タイトルを抜く
with open("train.txt", "r") as f_r_1:
    L = f_r_1.readlines()
    for i in range(len(L)):
        L[i] = L[i][2:len(L[i])-1] #ラベルと最後の改行記号を落とす
    tokens_train = tokenizer.batch_encode_plus(L, max_length = 20, pad_to_max_length=True, truncation=True) #truncation->長い入力データを、max_lenでカットする
with open("valid.txt", "r") as f_r_2:
    L = f_r_2.readlines()
    for i in range(len(L)):
        L[i] = L[i][2:len(L[i])-1]
    tokens_valid = tokenizer.batch_encode_plus(L, max_length = 20, pad_to_max_length=True, truncation=True)
with open("test.txt", "r") as f_r_3:
    L = f_r_3.readlines()
    for i in range(len(L)):
        L[i] = L[i][2:len(L[i])-1]
    tokens_test = tokenizer.batch_encode_plus(L, max_length = 20, pad_to_max_length=True, truncation=True)

#データをテンソル型にする
train_titles = torch.tensor(tokens_train["input_ids"])
train_mask = torch.tensor(tokens_train["attention_mask"])
train_labels = torch.tensor(np.load("train_label.npy"))

valid_titles = torch.tensor(tokens_valid["input_ids"])
valid_mask = torch.tensor(tokens_valid["attention_mask"])
valid_labels = torch.tensor(np.load("valid_label.npy"))

test_titles = torch.tensor(tokens_test["input_ids"])
test_mask = torch.tensor(tokens_test["attention_mask"])
test_labels = torch.tensor(np.load("test_label.npy"))

#訓練のためにデータを小分けにする
batch_size = 128
train_data = TensorDataset(train_titles, train_mask, train_labels) #三つの情報を一つのTensorDatasetオブジェクトにまとめる
train_sampler = RandomSampler(train_data) #SamplerはDataLoaderの引数になる。バッチサイズ分のインデックスを返す。
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size) #バッチごとにデータをまとめるのがDataLoader

valid_data = TensorDataset(valid_titles, valid_mask, valid_labels)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler = valid_sampler, batch_size = batch_size)

for param in bert.parameters(): #BERT自体のパラメータは変化させない（classifierのパラメータのみ動かす）
    param.requires_grad = False

#カテゴリ分類モデルの定義
class TextClassifierWithBERT(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 4)
        self.softmax = nn.Softmax()

    def forward(self, converted_titles, masks):
        #https://stackoverflow.com/questions/65132144/bertmodel-transformers-outputs-string-instead-of-tensor/65137768#65137768
        #return_dictをfalseにしないと、dictのkey（string）である"last_hidden_state", "pooler_output"が返ってきてしまう
        #pooler_output(文章の平均みたいなやつ)が欲しいのでそれだけもらってくる
        last_hidden_state, pooler_output = self.bert(converted_titles, attention_mask=masks, return_dict=False)
        x = self.fc1(pooler_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x

class_weights = compute_class_weight(class_weight = "balanced", classes = np.array([0,1,2,3]), y = np.array(train_labels))
weights = torch.tensor(class_weights, dtype = torch.float)
cross_entropy  = nn.CrossEntropyLoss(weight = weights)

classifier_model = TextClassifierWithBERT(bert)
optimizer = AdamW(classifier_model.parameters(), lr = 1e-3) #lrはデフォで1e-3
epochs = 10

#モデルの訓練
def train():
    classifier_model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and step != 0:
            print(f"  Batch {step}  of  {len(train_dataloader)}")

        classifier_model.zero_grad()
        train_titles, train_mask, train_labels = batch #TensorDatasetを元に戻す
        predictions = classifier_model(train_titles, train_mask)
        loss = cross_entropy(predictions, train_labels)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
        total_preds.append(predictions.detach().numpy()) #concatenateの操作をするために勾配情報を消す

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis = 0)

    return avg_loss, total_preds

def evaluate():
    print("\nEvaluating...")
    classifier_model.eval() # deactivate dropout layers
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(valid_dataloader):
        valid_titles, valid_mask, valid_labels = batch

        with torch.no_grad(): #deactivate autograd
            predictions = classifier_model(valid_titles, valid_mask)
            loss = cross_entropy(predictions, valid_labels)
            total_loss += float(loss)
            total_preds.append(predictions.detach().numpy())

    avg_loss = total_loss / len(valid_dataloader)
    total_preds  = np.concatenate(total_preds, axis = 0)

    return avg_loss, total_preds

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
best_valid_loss = float('inf')

#for each epoch
for epoch in range(epochs):
    print(f"\n Epoch {epoch + 1} / {epochs}")

    train_loss, _ = train()
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(classifier_model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


#classifier_model.load_state_dict(torch.load("saved_weights.pt"))

with torch.no_grad():
    prediction_for_test = classifier_model(test_titles, test_mask)
    predicted_labels = np.argmax(prediction_for_test.detach().numpy(), axis = 1)

print(classification_report(test_labels, predicted_labels))
