import unittest

import torch
from torch import nn

from experiment import ARMS, split_clips, scenarios, student_model, teacher_passes, training_loss


class PilotTest(unittest.TestCase):
    def test_matching_endpoint_labels_in_all_arms(self):
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(3)
        x0, x1 = torch.tensor([[1.0]]), torch.tensor([[4.0]])
        for arm in ARMS:
            loss = training_loss(model, arm, x0, x1, model(x0).detach(), model(x1).detach())
            self.assertAlmostEqual(loss.item(), 0)
            bad_right = training_loss(model, arm, x0, x1, model(x0).detach(), model(x1).detach() + 2)
            self.assertAlmostEqual(bad_right.item(), 2 if arm == "bc" else 6)

    def test_hoas_backprop_matches_parameter_finite_difference(self):
        torch.manual_seed(1)
        model = student_model(3, 2).double()
        x0, x1 = (torch.randn(4, 3, dtype=torch.double) for _ in range(2))
        a0, a1 = (torch.randn(4, 2, dtype=torch.double) for _ in range(2))
        def objective():
            return training_loss(model, "hoas", x0, x1, a0, a1) - training_loss(model, "bc", x0, x1, a0, a1)
        objective().backward()
        weight = model[0].weight
        derivative = weight.grad[0, 0].item()
        with torch.no_grad():
            weight[0, 0] += 1e-5
            upper = objective().item()
            weight[0, 0] -= 2e-5
            lower = objective().item()
        self.assertGreater(abs(derivative), 1e-8)
        self.assertAlmostEqual(derivative, (upper - lower) / 2e-5, places=7)

    def test_split_and_gate(self):
        train, test = split_clips(["walk1_subject1", "run1_subject5", "jump_subject1"])
        self.assertEqual(train, ["walk1_subject1"])
        self.assertEqual(test, ["run1_subject5"])
        self.assertTrue(set(c.kick_x for c in scenarios(train, True)) == {0})
        self.assertTrue(any(c.kick_x != 0 for c in scenarios(test, False)))
        summary = {"nominal": {"recovered": 9, "n": 10}, "disturbed": {"recovered": 3, "n": 4}}
        self.assertTrue(teacher_passes(summary))
        summary["disturbed"]["recovered"] -= 1
        self.assertFalse(teacher_passes(summary))


if __name__ == "__main__":
    unittest.main()
