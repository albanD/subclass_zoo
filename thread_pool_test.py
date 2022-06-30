import torch
from typing import List
import torch.nn.functional as F
import concurrent.futures as futures
D = 16
input_tensors = [torch.randn(D, device = 'cuda') for _ in range(8)]

def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

weight = torch.randn(D, D, device = 'cuda')
bias = torch.randn(D, device = 'cuda')

threadpool = futures.ThreadPoolExecutor(max_workers = 8)

future_res:List[futures.Future] = []

for i in range(8):
    future_res.append(threadpool.submit(predict, weight, bias, input_tensors[i]))

outs = [future_res[i].result() for i in range(8)]

print(type(outs[0]))
print(outs)