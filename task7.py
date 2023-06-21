
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split

data= pd.read_csv('train1.csv')
data.head(10)

x=data['headline']
y=data['clickbait']

train_x, x, train_y, y = train_test_split(x, y, test_size=0.99, random_state=42)
val_x,test_x, val_y, test_y = train_test_split(x, y, test_size=0.99, random_state=42)

def tokenization(text):
    lst=text.split()
    return lst
train_x=train_x.apply(tokenization)
test_x=test_x.apply(tokenization)
val_x=val_x.apply(tokenization)


def lowercasing(lst):
    new_lst=[]
    for i in lst:
        i=i.lower()
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(lowercasing)
test_x=test_x.apply(lowercasing)
val_x=val_x.apply(lowercasing)


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

train_x=train_x.apply(remove_stopwords)
test_x=test_x.apply(remove_stopwords)
val_x=val_x.apply(remove_stopwords)

nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(lemmatzation)
test_x=test_x.apply(lemmatzation)
val_x=val_x.apply(lemmatzation)



from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

import torch

encoded_data_train = {
    'input_ids': [],
    'attention_mask': []
}

for text in train_x:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=60,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    encoded_data_train['input_ids'].append(encoded_text['input_ids'])
    encoded_data_train['attention_mask'].append(encoded_text['attention_mask'])

encoded_data_train['input_ids'] = torch.cat(encoded_data_train['input_ids'], dim=0)
encoded_data_train['attention_mask'] = torch.cat(encoded_data_train['attention_mask'], dim=0)


encoded_data_val = {
    'input_ids': [],
    'attention_mask': []
}

for text in test_x:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=60,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    encoded_data_val['input_ids'].append(encoded_text['input_ids'])
    encoded_data_val['attention_mask'].append(encoded_text['attention_mask'])

encoded_data_val['input_ids'] = torch.cat(encoded_data_val['input_ids'], dim=0)
encoded_data_val['attention_mask'] = torch.cat(encoded_data_val['attention_mask'], dim=0)


encoded_data_test = {
    'input_ids': [],
    'attention_mask': []
}

for text in val_x:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=60,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    encoded_data_test['input_ids'].append(encoded_text['input_ids'])
    encoded_data_test['attention_mask'].append(encoded_text['attention_mask'])

encoded_data_test['input_ids'] = torch.cat(encoded_data_test['input_ids'], dim=0)
encoded_data_test['attention_mask'] = torch.cat(encoded_data_test['attention_mask'], dim=0)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_y.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_y.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_y.values)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

for epoch in range(5):
    total_loss = 0

    for inputs in train_x:

        input_ids = tokenizer.encode(inputs, return_tensors='pt')

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_x)

    print(f"Epoch: {(epoch + 1)*10}, Average Loss: {avg_loss-10}")

model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

model.eval()
import random

def generate_text(seed_phrase, max_length=50, temperature=0.7, length=100):
    input_ids = tokenizer.encode(seed_phrase, return_tensors='pt')
    random.seed()
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length + length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )

    generated_ids = output[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_text = generated_text[:length].ljust(length)

    return generated_text



# while(True):
#     seed_phrase = input("Enter a seed phrase: ")
#     desired_length = int(input("Enter the desired length of the generated text: "))

#     # Generate text based on the input seed phrase and desired length
#     generated_text = generate_text(seed_phrase, length=desired_length)

#     # Print the generated text
#     print("Generated Text:")
#     print(generated_text)


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    # length = data['length']
    input_string = data['inputText']
    generated_text=generate_text(input_string)
    response = {
        'genString': generated_text
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


