from functools import partial
import unittest

import dice_mc.torch as dice
import torch
from torch.testing import assert_close

assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


class TestMagicBox(unittest.TestCase):
    def test_returns_one(self):
        """Test that magic_box() returns 1"""
        self.assertEqual(dice.magic_box(torch.tensor(5.0)), 1.0)

    def test_derivatives(self):
        """Test that the derivatives of inner() are https://oeis.org/A000085"""

        def inner(tau):
            return dice.magic_box(tau**2 / 2).sum()

        fun = inner
        result = []
        for _ in range(10):
            result.append(fun(torch.tensor(1.0)))
            fun = torch.func.grad(fun)
        result = torch.stack(result)
        expected = torch.tensor([1, 1, 2, 4, 10, 26, 76, 232, 764, 2620], dtype=torch.float32)
        assert_close(result, expected)


class TestLeftSumToSize(unittest.TestCase):
    def test_unchanged(self):
        """Test that left_sum_to_size() returns the same tensor when the size is already correct"""
        x = torch.randn(3, 4, 5)
        assert_equal(dice.left_sum_to_size(x, (3, 4, 5)), x)

    def test_summed(self):
        """Test that left_sum_to_size() sums the last dimension when the size is smaller"""
        x = torch.randn(3, 4, 5)
        assert_close(dice.left_sum_to_size(x, (3, 4)), x.sum(-1))

    def test_summed_length_one(self):
        """Test that left_sum_to_size() sums following the broadcasting rules when the size is
        smaller and the last dimension is 1"""
        x = torch.randn(3, 4, 5)
        assert_close(dice.left_sum_to_size(x, (3, 1)), x.sum(-1).sum(-1, keepdim=True))

    def test_invalid(self):
        """Test that left_sum_to_size() raises an error when the size is too large"""
        x = torch.randn(3, 4, 5)
        with self.assertRaisesRegex(RuntimeError, "is not expandable"):
            dice.left_sum_to_size(x, (3, 4, 5, 6))


class TestLogPCategorical(unittest.TestCase):
    def test_logp(self):
        """Test that logp_categorical() computes the logprobs of the given actions"""
        probs = torch.tensor([0.25, 0.75])
        actions = torch.tensor(0)
        assert_close(dice.logp_categorical(probs.log(), actions).exp(), probs[0])

    def test_batch_logp(self):
        """Test that logp_categorical() computes the logprobs of the given batch of actions"""
        probs = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
        actions = torch.tensor([0, 1])
        assert_close(dice.logp_categorical(probs.log(), actions).exp(), probs[range(2), actions])


class TestSampleCategorical(unittest.TestCase):
    def test_sample(self):
        """Test that sample_categorical() samples from the given distribution"""
        probs = torch.tensor([0.25, 0.75]).expand(10000, -1)
        logp, actions = dice.sample_categorical(probs.log())
        sampled_probs = (actions == torch.arange(2)[:, None]).mean(dim=-1, dtype=torch.float32)
        assert_close(sampled_probs, probs[0], atol=0.01, rtol=0.01)

    def test_logp(self):
        """Test that sample_categorical() returns the logprobs of the given actions"""
        probs = torch.tensor([0.25, 0.75])
        logp, actions = dice.sample_categorical(probs.log())
        expected = probs.gather(-1, actions[..., None])[..., 0]
        assert_close(logp.exp(), expected)

    def test_batch_logp(self):
        """Test that sample_categorical() reports the logprobs of the given batch of actions"""
        probs = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
        logp, actions = dice.sample_categorical(probs.log())
        expected = probs.gather(-1, actions[..., None])[..., 0]
        assert_close(logp.exp(), expected)
