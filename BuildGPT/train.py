from traceback import print_tb

with open('../input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("Length of dataset in characters: ", len(text))
print(text[:1000])
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}#enumerate 함수는 인덱스와 값을 각각 출력. 앞에 ch:i는 딕셔너리 저장 형대. {ch : 'i'} 의 인코더 형대
itos = { i:ch for i,ch in enumerate(chars)}#디코더여서 integer to string
encode = lambda s: [stoi[c] for c in s]# encoder: take a string, output a list od integers
decode = lambda l: ''.join([itos[i] for i in l])# decoder: take a list od integers, output a string

print(encode('hii there'))
print(decode(encode('hii there')))

#let's now encode the entire text dataset and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])# the 1000 characters we looked at easier will to the GPT look like this

#let's now split up the data into train and validation sets
n = int(0.9*len(data))#firts 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('input')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):# batch dimension
    for t in range(block_size):# time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")