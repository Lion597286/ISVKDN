
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from stage2_train_student import student_training_step
from metric.loss import SoftTarget
from uestc_semantic_mapping import sgsm_semantic_slope


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.head = torch.nn.Linear(4, 3)
        self.seen_sizes = []

    def forward(self, inputs, return_features=None):
        self.seen_sizes.append(len(inputs))
        features = self.conv(inputs)
        semantic = features.mean(dim=(2, 3))
        if return_features is False:
            return self.head(semantic)
        return (features,) * 5 + (semantic, self.head(semantic))


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyStudent()
        self.seen_sizes = []

    def forward(self, gyro, acc, return_features):
        self.seen_sizes.append((len(gyro), len(acc)))
        g = self.encoder(gyro)
        a = self.encoder(acc)
        return tuple(value for pair in zip(g[:5], a[:5]) for value in pair) + (
            g[5], a[5], g[5], a[5], g[6], a[6])


class GraphControlChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_identity_graph_equals_bypass(self):
        logits = torch.randn(6, 3)
        embeddings = torch.randn(3, 8)
        with patch('uestc_semantic_mapping.lle_sparse_graph',
                   side_effect=lambda values, k: torch.eye(len(values))) as graph:
            graph_result = sgsm_semantic_slope(logits, embeddings, k=2)
            self.assertEqual(graph.call_count, 2)
            graph.reset_mock()
            bypass = sgsm_semantic_slope(logits, embeddings, k=2, propagate_graph=False)
            graph.assert_not_called()
        torch.testing.assert_close(graph_result, bypass)

    def test_complete_no_graph_step_keeps_masked_inputs_and_updates_student(self):
        student = TinyStudent()
        teacher = TinyTeacher().requires_grad_(False).eval()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        args = SimpleNamespace(sgsm_mode='no_graph', sgsm_k=2, soft_kd_weight=1,
                               feature_weight=1, semantic_weight=1)
        batch = (torch.randn(3, 3, 4, 4), torch.randn(3, 3, 4, 4),
                 torch.randn(3, 3, 4, 4), torch.tensor([0, 1, 2]))
        before = {name: value.detach().clone() for name, value in student.named_parameters()}
        with patch('uestc_semantic_mapping.lle_sparse_graph',
                   side_effect=AssertionError('no_graph must not build a graph')):
            logits, losses = student_training_step(student, teacher, batch, torch.randn(3, 8),
                args, optimizer, scaler, False,
                (torch.nn.CrossEntropyLoss(), SoftTarget(4), torch.nn.MSELoss()))
        self.assertEqual(student.seen_sizes, [6])
        self.assertEqual(teacher.seen_sizes, [(6, 6)])
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertTrue(all(torch.isfinite(value).item() for value in losses.values()))
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in student.named_parameters()))
        self.assertTrue(all(value.grad is None for value in teacher.parameters()))

    def test_video_baseline_remains_teacher_free(self):
        student = TinyStudent()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        args = SimpleNamespace(sgsm_mode='off', sgsm_k=2, soft_kd_weight=0,
                               feature_weight=0, semantic_weight=0)
        with patch('stage2_train_student.sgsm_semantic_slope',
                   side_effect=AssertionError('baseline must not use SGSM')):
            logits, losses = student_training_step(student, None,
                (torch.randn(3, 3, 4, 4), torch.tensor([0, 1, 2])), None, args,
                optimizer, torch.cuda.amp.GradScaler(enabled=False), False,
                (torch.nn.CrossEntropyLoss(), SoftTarget(4), torch.nn.MSELoss()))
        self.assertEqual(student.seen_sizes, [3])
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertEqual(losses['loss'].item(), losses['ce'].item())
        self.assertTrue(all(losses[key].item() == 0 for key in ('soft_kd', 'feature', 'semantic')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
