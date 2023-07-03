# dice-mc

Implements [DiCE: The Infinitely Differentiable Monte-Carlo Estimator](https://arxiv.org/abs/1802.05098) in PyTorch.

DiCE is a surrogate loss for the score function estimator (REINFORCE), an unbiased Monte Carlo estimator of the gradient and higher-order derivatives (Hessian etc.) of the expectation of the loss, where the computation graph contains nondifferentiable stochastic nodes, such as sampling from a categorical distribution. An example use case of DiCE is fine-tuning a large language model with a loss that depends on outputs sampled from it during training, as is done in [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback).

## Usage

```python
import dice_mc.torch as dice
```

Stochastic nodes are represented as a tensor of the logprobs of the sampled actions. `logp_categorical()` and `sample_categorical()` create and return stochastic nodes for categorical distributions. `logp_categorical()` is used when you already have the actions that were sampled, and `sample_categorical()` is used when you need to sample actions from the logits.

```python
logp_1, actions_1 = dice.sample_categorical(logits_1)
```

Cost nodes are represented as a tensor of DiCE surrogate losses. `cost_node()` creates and returns a cost node given a cost tensor and an iterable of stochastic nodes that the costs were causally influenced by. (Including additional stochastic nodes will increase the variance of the estimator, but will not introduce bias.)

```python
losses_1 = dice.cost_node(losses_1, [logp_1])
```

The output of `cost_node()` can be differentiated to propagate gradients to the stochastic nodes given in the second argument. DiCE surrogate losses, when autodifferentiated, produce correct Monte Carlo estimators of higher order derivatives as well. The forward pass values of the DiCE surrogate losses are not modified, so they can be printed or used in metrics.

```python
losses = losses_1 + 0.1 * losses_2
loss = losses.mean()
loss.backward()
opt.step()
```

Baselines can be used to reduce the variance of the estimator. DiCE baseline terms are scalars with the value of zero that have the gradient of baseline subtraction. `EMABaseline` is a simple exponential moving average baseline. `EMABaseline` contains state which should be saved and loaded when checkpointing.

```python
baseline = dice.EMABaseline().to(device)
...
losses = losses + baseline(losses, [logp_1, logp_2, logp_3])  # All stochastic nodes
loss = losses.mean()
loss.backward()
opt.step()
```

If you have batches of losses, you can use `batch_baseline_term()`, a DiCE version of the [REINFORCE with replacement baseline](https://openreview.net/forum?id=r1lgTGL5DE). It uses the mean of the other losses in the batch as the baseline for each loss in the batch.

### A note on batching

If you are computing samples in batches where each batch item is independent of the others, you should provide a 1D tensor of losses, one per batch item, to `cost_node()` and only afterward take the mean along the batch dimension. This will result in a lower variance estimator whose variance decreases as you increase the batch size. This is because each stochastic node is secretly a batch of stochastic nodes, one per batch item, and each cost node is secretly a batch of cost nodes, one per batch item, and the cost nodes only depend on stochastic nodes with the same batch index, so they can be excluded to reduce variance.

### Making a stochastic node from a generation from a language model

You can sample from an autoregressive language model and then, after the fact, create a stochastic node from the logits and the sampled actions.

Note: for Hugging Face models, top-k is 50 by default which will make the sampled tokens diverge from the distribution given by the logits. You should set top-k to 0.

```python
tokens = model.generate(..., do_sample=True, temperature=1.0, top_k=0)
```

After sampling, you can run the tokens through the model once with gradients enabled to get logits which require grad, and create a stochastic node from the logits and tokens:

```python
outputs = model(tokens, attention_mask=attention_mask)
logp = dice.logp_categorical(outputs.logits[:, prompt_len - 1 : -1], tokens[:, prompt_len:])
```

The prompt should be excluded except for the logits for the last prompt token. The tokens should be shifted one position left so that each token lines up with the vector of logits it was sampled from. The log probability under the model of each *prefix* of the tokens (subsequence that contains the first token) is given by summing the log probabilities of the tokens in the prefix, so take `logp.cumsum(dim=1)` to get the log probability of each prefix. The "stochastic node" created by `logp_categorical()`, for an autoregressive sequence model, is secretly a sequence of stochastic nodes where each node is only causally influenced by nodes to its left. Again, you can use this fact to reduce the variance of the estimator: suppose one of your loss terms is the KL divergence from the logits of some other model. The KL penalty for a given token is not affected by tokens to its right, so they can be excluded:

```python
losses_kl = F.kl_div(
    F.log_softmax(outputs_old_model.logits[:, prompt_len:], dim=-1),
    F.log_softmax(outputs.logits[:, prompt_len:], dim=-1),
    reduction="none",
    log_target=True,
).sum(dim=-1)
logp_cumsum = torch.cumsum(logp, dim=1)
losses_kl = dice.cost_node(losses_kl, [logp_cumsum])
```

If you are using a batch size greater than 1, this snippet will also correctly follow the batching advice above.
