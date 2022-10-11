#!/usr/bin/env python3
import torch
if torch.cuda.is_available():
  print("cuda is available, supported platforms:")
  print(torch.cuda.get_arch_list())
else:
  print("cuda not available")
x = torch.rand(5, 3)
print(x)
