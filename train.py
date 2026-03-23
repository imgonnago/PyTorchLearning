with open('input.txt','r',encoding='utf-8') as f:
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
