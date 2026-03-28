# Bug List — QANet Codebase Debugging

All 37 intentional bugs identified and fixed. Categorised by file.

---

## Stage I: Pipeline Bugs (Runtime-Breaking)

### 1. `TrainTools/train.py` — Namespace construction syntax error
- **Bug:** `argparse.Namespace({k: v ...})` — passed a dict as a positional argument
- **Fix:** `argparse.Namespace(**{k: v ...})` — must unpack with `**`
- **Impact:** Crashes immediately on launch with `TypeError`; training cannot start at all

### 2. `TrainTools/train_utils.py` — Called `.item()` before `.backward()`
- **Bug:** `loss.item().backward()` — `.item()` returns a Python float, which has no `.backward()` method
- **Fix:** `loss.backward()`
- **Impact:** `AttributeError` on every training step; backpropagation never runs

### 3. `TrainTools/train_utils.py` — Gradient clipping after optimizer step
- **Bug:** `optimizer.step()` was called before `clip_grad_norm_`, then `scheduler.step()` was missing
- **Fix:** Call `loss.backward()` → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()` in correct order
- **Impact:** Gradients are clipped after weights are already updated, making clipping ineffective; unclipped gradients can cause instability

### 4. `Schedulers/scheduler.py` — Missing `"none"` key in scheduler registry
- **Bug:** The registry did not include `"none"` as a valid scheduler name
- **Fix:** Added `"none": lambda_scheduler` to the registry dict
- **Impact:** `ValueError` when `scheduler_name="none"` is passed; the default training configuration crashes

---

## Stage II: Deep Learning Mechanism Bugs

### Loss Function

### 5. `Losses/loss.py` — Swapped arguments in `nll_loss`
- **Bug:** `F.nll_loss(y1, p1)` — arguments reversed (correct: input first, target second)
- **Fix:** `F.nll_loss(p1, y1)`
- **Impact:** Loss computes nonsense values; training signal is completely wrong

---

### Activations

### 6. `Models/Activations/relu.py` — ReLU clips wrong side
- **Bug:** `x.clamp(max=0.0)` — keeps only negative values
- **Fix:** `x.clamp(min=0.0)` — keeps only non-negative values
- **Impact:** ReLU outputs the inverse of the correct activation; all positive activations are zeroed out, destroying feature learning

### 7. `Models/Activations/leakeyReLU.py` — LeakyReLU applies slope to wrong branch
- **Bug:** `torch.where(x < 0, x, self.negative_slope * x)` — applies slope to positive values, passes negatives unchanged
- **Fix:** `torch.where(x < 0, self.negative_slope * x, x)`
- **Impact:** Positive activations are shrunk, negative activations pass through at full scale — the exact opposite of LeakyReLU

---

### Initializations

### 8. `Models/Initializations/kaiming.py` — Wrong constant in Kaiming Normal
- **Bug:** `std = math.sqrt(1.0 / fan)` — uses constant 1.0 instead of 2.0
- **Fix:** `std = math.sqrt(2.0 / fan)`
- **Impact:** Variance is half the intended value; activations shrink layer-by-layer, causing vanishing gradients

### 9. `Models/Initializations/kaiming.py` — Wrong constant in Kaiming Uniform
- **Bug:** Same as above: `std = math.sqrt(1.0 / fan)`
- **Fix:** `std = math.sqrt(2.0 / fan)`
- **Impact:** Same as Bug 8 — uniform variant also initialises with incorrect scale

### 10. `Models/Initializations/xavier.py` — Xavier Normal uses product instead of sum
- **Bug:** `std = gain * math.sqrt(2.0 / (fan_in * fan_out))` — multiplies fans instead of adding
- **Fix:** `std = gain * math.sqrt(2.0 / (fan_in + fan_out))`
- **Impact:** For typical layer sizes, std is orders of magnitude too small, causing extreme vanishing gradients

### 11. `Models/Initializations/xavier.py` — Xavier Uniform uses product instead of sum
- **Bug:** Same as Bug 10 for the uniform variant
- **Fix:** `std = gain * math.sqrt(2.0 / (fan_in + fan_out))`
- **Impact:** Same as Bug 10

---

### Normalizations

### 12. `Models/Normalizations/layernorm.py` — `keepdim=False` causes shape mismatch
- **Bug:** `mean = x.mean(dim=dims, keepdim=False)` — collapses dimensions, making subtraction `x - mean` broadcast incorrectly
- **Fix:** `keepdim=True`
- **Impact:** Shape mismatch in normalization; incorrect mean subtraction corrupts all normalized activations

### 13. `Models/Normalizations/layernorm.py` — `keepdim=False` on variance
- **Bug:** Same issue for variance: `x.var(dim=dims, keepdim=False, ...)`
- **Fix:** `keepdim=True`
- **Impact:** Same as Bug 12 — variance has wrong shape for division

### 14. `Models/Normalizations/layernorm.py` — Weight and bias swapped in affine transform
- **Bug:** `return x_norm * self.bias + self.weight` — applies bias as scale and weight as shift
- **Fix:** `return x_norm * self.weight + self.bias`
- **Impact:** Scale and shift parameters are reversed; the learned affine transformation is incorrect

### 15. `Models/Normalizations/groupnorm.py` — Groups and channels swapped in reshape
- **Bug:** `x.view(B, C // self.G, self.G, *spatial)` — puts channel-per-group first, num-groups second
- **Fix:** `x.view(B, self.G, C // self.G, *spatial)`
- **Impact:** Normalisation statistics are computed across the wrong elements; the grouping is completely incorrect

---

### Convolutions

### 16. `Models/conv.py` — `unfold` applied to wrong dimension in Conv1d
- **Bug:** `x.unfold(1, self.kernel_size, 1)` — unfolds along channel dim (dim 1) instead of length dim (dim 2)
- **Fix:** `x.unfold(2, self.kernel_size, 1)`
- **Impact:** The sliding window operates on channels instead of positions; convolution output is completely wrong

### 17. `Models/conv.py` — Padding height used for width pad tensor in Conv2d
- **Bug:** `pad_w = x.new_zeros(B, C_in, H, p)` — uses original `H` after height padding has changed it
- **Fix:** `pad_w = x.new_zeros(B, C_in, x.size(2), p)` — uses updated height after padding
- **Impact:** Width padding tensor has wrong height dimension, causing shape mismatch when concatenating

### 18. `Models/conv.py` — Depthwise and pointwise convolutions applied in wrong order
- **Bug:** `return self.depthwise_conv(self.pointwise_conv(x))` — pointwise first
- **Fix:** `return self.pointwise_conv(self.depthwise_conv(x))` — depthwise first
- **Impact:** Depthwise separable convolution is not equivalent to the intended factored operation; wrong channel mixing

---

### Dropout

### 19. `Models/dropout.py` — Dropout scales by `p` instead of `(1 - p)`
- **Bug:** `return x * mask / self.p` — divides by drop probability
- **Fix:** `return x * mask / (1.0 - self.p)` — divides by keep probability
- **Impact:** Inverted-dropout scaling is wrong; expected value of activations at test time is incorrect, scaling all outputs by a constant factor

---

### Embedding

### 20. `Models/embedding.py` — Highway network transposes wrong dimensions
- **Bug:** `x = x.transpose(0, 2)` — swaps batch dim with length dim
- **Fix:** `x = x.transpose(1, 2)` — swaps channel dim with length dim (correct for linear layers)
- **Impact:** Batch and length dimensions are swapped; linear layers receive tensors of wrong shape

### 21. `Models/embedding.py` — Character embedding permuted to wrong layout
- **Bug:** `ch_emb.permute(0, 2, 1, 3)` — results in `[B, L, d_char, char_len]`
- **Fix:** `ch_emb.permute(0, 3, 1, 2)` — results in `[B, d_char, L, char_len]` (required by Conv2d)
- **Impact:** Channels and length dimensions are swapped; Conv2d receives wrong tensor layout

---

### Encoder

### 22. `Models/encoder.py` — Positional encoding frequency tensor unsqueezed on wrong dim
- **Bug:** `freqs.unsqueeze(0)` — shape becomes `[1, d_model]` instead of `[d_model, 1]`
- **Fix:** `freqs.unsqueeze(1)` — shape is `[d_model, 1]` for correct broadcasting with position indices
- **Impact:** Positional encodings are computed incorrectly; position information is wrong

### 23. `Models/encoder.py` — Multi-head attention output permuted incorrectly
- **Bug:** `out.permute(1, 2, 0, 3)` — swaps batch and head dimensions
- **Fix:** `out.permute(0, 2, 1, 3)` — correctly moves head dim so batch remains first
- **Impact:** Batch and head dimensions are mixed; attention outputs are scrambled across examples

### 24. `Models/encoder.py` — Wrong norm layer index in conv stack
- **Bug:** `out = self.norms[i + 1](out)` — uses next norm layer, skipping norm 0 and going out of bounds
- **Fix:** `out = self.norms[i](out)` — uses the correct norm for each conv layer
- **Impact:** Off-by-one means wrong normalization is applied to each conv; last iteration causes `IndexError`

### 25. `Models/encoder.py` — Residual connection discards attention output
- **Bug:** `out = res` — overwrites attention output with the pre-attention residual
- **Fix:** `out = out + res` — adds residual to attention output
- **Impact:** The self-attention layer has zero effect; all attention computation is discarded

---

### Attention

### 26. `Models/attention.py` — Context-query attention matrix multiplication order wrong
- **Bug:** `A = torch.bmm(Q, S1)` — multiplies query by attention weights
- **Fix:** `A = torch.bmm(S1, Q)` — multiplies attention weights by query (correct: `[B, Lc, Lq] x [B, Lq, C]`)
- **Impact:** Matrix dimensions don't align for batched matmul; `RuntimeError` or wrong attended context

---

### Output Head

### 27. `Models/heads.py` — Pointer head concatenates on wrong dimension
- **Bug:** `torch.cat([M1, M2], dim=0)` — concatenates along batch dimension
- **Fix:** `torch.cat([M1, M2], dim=1)` — concatenates along channel dimension
- **Impact:** Batch size doubles instead of channel size doubling; subsequent matmul has wrong shape

---

### QANet Forward Pass

### 28. `Models/qanet.py` — Word and char embeddings swapped
- **Bug:** `Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)` — word IDs passed to char embedding and vice versa
- **Fix:** `Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)`
- **Impact:** Embedding lookup uses wrong vocabulary sizes; likely `IndexError` or produces completely wrong embeddings

### 29. `Models/qanet.py` — Context and query masks swapped in CQ attention
- **Bug:** `self.cq_att(Ce, Qe, qmask, cmask)` — masks passed in reversed order
- **Fix:** `self.cq_att(Ce, Qe, cmask, qmask)` — context mask for context, query mask for query
- **Impact:** Padding is masked in the wrong sequence; attention attends to padding positions and ignores real tokens

---

### Optimizers

### 30. `Optimizers/adam.py` — Weight decay subtracts instead of adds gradient
- **Bug:** `grad = grad.add(p, alpha=-wd)` — negative alpha subtracts the L2 penalty
- **Fix:** `grad = grad.add(p, alpha=wd)` — L2 regularisation must add the parameter to the gradient
- **Impact:** Weight decay acts as weight growth; parameters grow unboundedly instead of being regularised

### 31. `Optimizers/adam.py` — Wrong state dict keys accessed
- **Bug:** `m, v = state["m"], state["v"]` — keys don't exist (initialized as `"exp_avg"` and `"exp_avg_sq"`)
- **Fix:** `m, v = state["exp_avg"], state["exp_avg_sq"]`
- **Impact:** `KeyError` on every Adam step; optimizer crashes immediately

### 32. `Optimizers/adam.py` — Second moment updated with gradient instead of gradient squared
- **Bug:** `v.mul_(beta2).add_(grad, alpha=1.0 - beta2)` — adds `grad` not `grad²`
- **Fix:** `v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)`
- **Impact:** Second moment estimate is wrong; Adam's adaptive scaling is based on gradient magnitude not variance

### 33. `Optimizers/adam.py` — Bias correction uses multiplication instead of exponentiation
- **Bug:** `bias_correction1 = 1.0 - beta1 * t` — linear instead of exponential decay
- **Fix:** `bias_correction1 = 1.0 - beta1 ** t`
- **Impact:** Bias correction is wrong at every step; early steps are not corrected properly, later steps may produce negative correction terms

### 34. `Optimizers/adam.py` — Same exponentiation bug for beta2 bias correction
- **Bug:** `bias_correction2 = 1.0 - beta2 * t`
- **Fix:** `bias_correction2 = 1.0 - beta2 ** t`
- **Impact:** Same as Bug 33 for the second moment estimate

### 35. `Optimizers/sgd.py` — Weight decay subtracts instead of adds gradient
- **Bug:** `grad = grad.add(p, alpha=-wd)`
- **Fix:** `grad = grad.add(p, alpha=wd)`
- **Impact:** L2 regularisation acts as L2 anti-regularisation; parameters grow rather than shrink

### 36. `Optimizers/sgd_momentum.py` — Velocity buffer stored under wrong key
- **Bug:** `state["vel"] = torch.zeros_like(p)` — initialized as `"vel"` but accessed as `"velocity"`
- **Fix:** `state["velocity"] = torch.zeros_like(p)`
- **Impact:** `KeyError` on every step after first; SGD with momentum crashes

### 37. `Optimizers/sgd_momentum.py` — Velocity update subtracts gradient instead of adding
- **Bug:** `v.mul_(mu).sub_(grad)` — subtracts gradient from velocity
- **Fix:** `v.mul_(mu).add_(grad)` — accumulates gradient into velocity
- **Impact:** Momentum accelerates in the opposite direction of the gradient; the optimizer diverges

---

### Schedulers

### 38. `Schedulers/cosine_scheduler.py` — Missing 0.5 factor and wrong `math.PI`
- **Bug:** `self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.PI * t / self.T_max))` — factor of 2 too large and `math.PI` doesn't exist
- **Fix:** `self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max))`
- **Impact:** `AttributeError` from `math.PI`; if somehow reached, LR would be double the intended value

### 39. `Schedulers/lambda_scheduler.py` — LR updated by addition instead of multiplication
- **Bug:** `return [base_lr + factor for base_lr in self.base_lrs]` — adds lambda factor
- **Fix:** `return [base_lr * factor for base_lr in self.base_lrs]` — multiplies by lambda factor
- **Impact:** LR schedule adds an offset rather than scaling; any constant lambda like 1.0 would add 1.0 to every LR, making the actual LR twice the intended value

### 40. `Schedulers/step_scheduler.py` — Gamma applied multiplicatively instead of as power
- **Bug:** `base_lr * self.gamma * (t // self.step_size)` — multiplies by step count
- **Fix:** `base_lr * self.gamma ** (t // self.step_size)` — raises gamma to the power of step count
- **Impact:** Instead of exponential decay, LR decays linearly and reaches zero then goes negative

---

## Summary Table

| # | File | Category | Type |
|---|------|----------|------|
| 1 | `TrainTools/train.py` | Pipeline | Syntax |
| 2 | `TrainTools/train_utils.py` | Pipeline | API misuse |
| 3 | `TrainTools/train_utils.py` | Pipeline | Wrong order |
| 4 | `Schedulers/scheduler.py` | Pipeline | Missing registry key |
| 5 | `Losses/loss.py` | Loss | Swapped args |
| 6 | `Models/Activations/relu.py` | Activation | Wrong clamp |
| 7 | `Models/Activations/leakeyReLU.py` | Activation | Wrong branch |
| 8 | `Models/Initializations/kaiming.py` | Init | Wrong constant (normal) |
| 9 | `Models/Initializations/kaiming.py` | Init | Wrong constant (uniform) |
| 10 | `Models/Initializations/xavier.py` | Init | Product vs sum (normal) |
| 11 | `Models/Initializations/xavier.py` | Init | Product vs sum (uniform) |
| 12 | `Models/Normalizations/layernorm.py` | Norm | keepdim mean |
| 13 | `Models/Normalizations/layernorm.py` | Norm | keepdim var |
| 14 | `Models/Normalizations/layernorm.py` | Norm | Swapped weight/bias |
| 15 | `Models/Normalizations/groupnorm.py` | Norm | Swapped group dims |
| 16 | `Models/conv.py` | Conv | Wrong unfold dim |
| 17 | `Models/conv.py` | Conv | Wrong pad height |
| 18 | `Models/conv.py` | Conv | Wrong conv order |
| 19 | `Models/dropout.py` | Dropout | Wrong scale factor |
| 20 | `Models/embedding.py` | Embedding | Wrong transpose dims |
| 21 | `Models/embedding.py` | Embedding | Wrong permute order |
| 22 | `Models/encoder.py` | Encoder | Wrong unsqueeze dim |
| 23 | `Models/encoder.py` | Encoder | Wrong permute order |
| 24 | `Models/encoder.py` | Encoder | Wrong norm index |
| 25 | `Models/encoder.py` | Encoder | Missing residual add |
| 26 | `Models/attention.py` | Attention | Wrong bmm order |
| 27 | `Models/heads.py` | Head | Wrong cat dim |
| 28 | `Models/qanet.py` | Model | Swapped embeddings |
| 29 | `Models/qanet.py` | Model | Swapped masks |
| 30 | `Optimizers/adam.py` | Optimizer | Wrong weight decay sign |
| 31 | `Optimizers/adam.py` | Optimizer | Wrong state keys |
| 32 | `Optimizers/adam.py` | Optimizer | grad vs grad² |
| 33 | `Optimizers/adam.py` | Optimizer | Mul vs pow (beta1) |
| 34 | `Optimizers/adam.py` | Optimizer | Mul vs pow (beta2) |
| 35 | `Optimizers/sgd.py` | Optimizer | Wrong weight decay sign |
| 36 | `Optimizers/sgd_momentum.py` | Optimizer | Wrong state key |
| 37 | `Optimizers/sgd_momentum.py` | Optimizer | Sub vs add gradient |
| 38 | `Schedulers/cosine_scheduler.py` | Scheduler | Missing 0.5 + math.PI |
| 39 | `Schedulers/lambda_scheduler.py` | Scheduler | Add vs multiply |
| 40 | `Schedulers/step_scheduler.py` | Scheduler | Multiply vs power |

**Total: 40 bugs identified and fixed.**
