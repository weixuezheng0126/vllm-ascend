"""Exercise upstream PP transport code on CPU with deterministic P2P queues.

Only the communication backend is mocked. Slicing, receive allocation,
postprocessing, sync wrappers, and per-key routing run the installed vLLM code.
"""

import ast
import importlib.util
import unittest
from collections import deque, namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from vllm_ascend.worker.v2.pp_transport import get_pp_topk_all_gather_overrides

KEY = "pp_transport_topk_indices_0"


def upstream_methods():
    spec = importlib.util.find_spec("vllm")
    root = (
        Path(next(iter(spec.submodule_search_locations))) if spec else Path(__file__).resolve().parents[5] / "vllm/vllm"
    )
    source = root / "distributed/parallel_state.py"
    tree = ast.parse(source.read_text())
    names = {"_should_use_all_gather", "isend_tensor_dict", "irecv_tensor_dict", "send_tensor_dict", "recv_tensor_dict"}
    group = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GroupCoordinator")
    nodes = [
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_split_tensor_dict")
    ]
    nodes += [node for node in group.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(
        torch=torch, Any=Any, deque=deque, TensorMetadata=namedtuple("TensorMetadata", "device dtype size")
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class Work:
    def wait(self):
        pass

    def is_completed(self):
        return True


class TestPPTopKTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = upstream_methods()

    def simulate(self, tp_size, rank_local, use_override, sync, gather=True, tokens=20):
        bus, metadata, received_ordinals = {}, {}, {}
        active = dict(tp=0, send_index=0, recv_index=0)
        gather_calls = []
        override = get_pp_topk_all_gather_overrides(rank_local) if use_override else None

        def send(tensor, **kwargs):
            bus[(active["tp"], active["send_index"])] = tensor.clone()
            active["send_index"] += 1
            return Work()

        def recv(tensor, **kwargs):
            index = active["recv_index"]
            tensor.copy_(bus[(active["tp"], index)])
            received_ordinals[id(tensor)] = index
            active["recv_index"] += 1
            return Work()

        def group(tp):
            def all_gather(tensor, dim):
                index = received_ordinals[id(tensor)]
                gather_calls.append((tp, index))
                return torch.cat([bus[(rank, index)] for rank in range(tp_size)], dim=dim)

            return SimpleNamespace(world_size=tp_size, rank_in_group=tp, all_gather=all_gather)

        def coordinator(tp):
            item = SimpleNamespace(
                world_size=2,
                rank_in_group=0,
                ranks=[0, 1],
                use_cpu_custom_send_recv=False,
                device_group=None,
                cpu_group=None,
                _reap_completed_isends=lambda: None,
            )

            def send_object(value, dst):
                metadata[tp] = value
                return Work()

            item.isend_object = send_object
            item.recv_object = lambda src: metadata[tp]
            for name in ("_should_use_all_gather", "isend_tensor_dict", "irecv_tensor_dict"):
                setattr(item, name, self.upstream[name].__get__(item))
            return item

        sent, actual = [], []
        with (
            patch.object(torch.distributed, "isend", side_effect=send),
            patch.object(torch.distributed, "irecv", side_effect=recv),
            patch.object(torch.distributed, "is_initialized", return_value=True),
        ):
            for tp in range(tp_size):
                active.update(tp=tp, send_index=0)
                topk = torch.full((tokens, 2048), -1, dtype=torch.int32)
                local = (tokens + tp_size - 1) // tp_size
                for row in range(min(local, tokens)):
                    topk[row, : min(tp * local + row + 1 if rank_local else row + 1, 2048)] = torch.arange(
                        min(tp * local + row + 1 if rank_local else row + 1, 2048)
                    )
                tensors = {
                    KEY: topk,
                    "hidden_states": torch.arange(tokens * 4).reshape(tokens, 4),
                    "residual": torch.ones(tokens, 4),
                    "pp_transport_aux_hidden_states_0": torch.ones(tokens, 4) * 3,
                    "unknown_future_key": torch.ones(tokens, 4) * 5,
                }
                sent.append(tensors)
                worker = coordinator(tp)
                method = self.upstream["send_tensor_dict" if sync else "isend_tensor_dict"]
                method(worker, tensors, all_gather_group=group(tp) if gather else None, all_gather_tensors=override)
            for tp in range(tp_size):
                active.update(tp=tp, recv_index=0)
                worker = coordinator(tp)
                method = self.upstream["recv_tensor_dict" if sync else "irecv_tensor_dict"]
                output = method(worker, all_gather_group=group(tp) if gather else None, all_gather_tensors=override)
                if sync:
                    tensors = output
                else:
                    tensors, handles, postprocess = output
                    for handle in handles:
                        handle.wait()
                    for fn in postprocess:
                        fn()
                actual.append(tensors)
        return sent, actual, gather_calls

    def test_baseline_reproduces_rank_local_prefix_corruption(self):
        for tp_size in (2, 4):
            sent, received, _ = self.simulate(tp_size, True, False, False)
            self.assertTrue(torch.equal(sent[0][KEY], received[1][KEY]))
            self.assertFalse(torch.equal(sent[1][KEY], received[1][KEY]))

    def test_override_preserves_each_rank_and_other_tensor_policy(self):
        for tp in (1, 2, 4):
            for rank_local in (False, True):
                for sync in (False, True):
                    for gather in (False, True):
                        for tokens in (20, 21):
                            with self.subTest(tp=tp, rank_local=rank_local, sync=sync, gather=gather, tokens=tokens):
                                sent, received, gathers = self.simulate(tp, rank_local, True, sync, gather, tokens)
                                for expected, actual in zip(sent, received):
                                    for key in expected:
                                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                                expected_gathers = tp * (4 if rank_local else 5) if gather else 0
                                self.assertEqual(len(gathers), expected_gathers)

    def test_worker_passes_identical_override_to_send_and_receive(self):
        source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
        tree = ast.parse(source.read_text())
        worker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
        method = next(
            node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
        )

        class Intermediate:
            def __init__(self, tensors, **kwargs):
                self.tensors = tensors
                self.kv_connector_output = None

        for rank_local in (False, True):
            calls = []
            group = SimpleNamespace(is_first_rank=False, is_last_rank=False)

            def receive(calls=calls, **kwargs):
                calls.append(("recv", kwargs))
                return {}, [], []

            def send(*args, calls=calls, **kwargs):
                calls.append(("send", kwargs))
                return []

            group.irecv_tensor_dict = receive
            group.isend_tensor_dict = send
            tp_group = object()
            namespace = dict(
                get_ascend_config=lambda rank_local=rank_local: SimpleNamespace(
                    msmonitor_use_daemon=False, enable_dsa_cp=rank_local
                ),
                get_pp_group=lambda group=group: group,
                get_tp_group=lambda tp_group=tp_group: tp_group,
                enable_sp=lambda: False,
                get_pp_topk_all_gather_overrides=get_pp_topk_all_gather_overrides,
                AsyncIntermediateTensors=Intermediate,
                IntermediateTensors=Intermediate,
                ModelRunnerOutput=type("Output", (), {}),
                AsyncModelRunnerOutput=type("AsyncOutput", (), {}),
                NoneType=type(None),
            )
            node = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(node), str(source), "exec"), namespace)
            worker = SimpleNamespace(
                log_memory_stats=lambda: None,
                _pp_send_work=[],
                profiler=None,
                model_runner=SimpleNamespace(execute_model=lambda *args: Intermediate({})),
                use_v2_model_runner=False,
                vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend="mp")),
            )
            namespace["execute_model"](worker, SimpleNamespace(total_num_scheduled_tokens=20))
            self.assertEqual([item[0] for item in calls], ["recv", "send"])
            for _, kwargs in calls:
                self.assertIs(kwargs["all_gather_group"], tp_group)
                self.assertEqual(kwargs["all_gather_tensors"], {KEY: False} if rank_local else None)

    def test_exact_key_override_and_default_policy(self):
        self.assertIsNone(get_pp_topk_all_gather_overrides(False))
        override = get_pp_topk_all_gather_overrides(True)
        self.assertEqual(override, {KEY: False})
        group = SimpleNamespace(world_size=4)
        route = self.upstream["_should_use_all_gather"]
        for key in ("hidden_states", "residual", "pp_transport_aux_hidden_states_0", "unknown_future_key"):
            self.assertTrue(route(None, key, 20, group, override))
            self.assertFalse(route(None, key, 21, group, override))
        self.assertFalse(route(None, KEY, 20, group, override))


if __name__ == "__main__":
    unittest.main()
