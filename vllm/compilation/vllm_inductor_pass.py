# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import operator
import time
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, TypeVar

import regex as re
import torch
from torch import fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.utils import Range

logger = init_logger(__name__)

_pass_context = None
P = ParamSpec("P")
R = TypeVar("R")


class PassContext:
    def __init__(self, compile_range: Range):
        self.compile_range: Range = compile_range


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(compile_range: Range) -> Generator[None, None, None]:
    """A context manager that stores the current pass context,
    usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(compile_range)
    try:
        yield
    finally:
        _pass_context = prev_context


@dataclass
class InductorCompilationConfig:
    splitting_ops: list[str] | None = None
    use_inductor_graph_partition: bool = False


class VllmInductorPass(CustomGraphPass):  # type: ignore[misc]
    """
    A custom graph pass for vLLM that uses a hash of its source as the UUID.
    It provides timing, logging, and dumping utilities, with optional access
    to vLLM PassConfig.
    """

    dump_prefix: ClassVar[int | None] = None
    """Keep track of pass index for debug dump ordering."""

    def __init__(self, config: VllmConfig | None = None):
        if config is not None:
            # Get only the necessary CompilationConfig for the inductor pass,
            # since full `CompilationConfig` contains pointer to model which
            # is unsafe.
            self.compilation_config: InductorCompilationConfig | None = (
                InductorCompilationConfig(
                    splitting_ops=config.compilation_config.splitting_ops,
                    use_inductor_graph_partition=config.compilation_config.use_inductor_graph_partition,
                )
            )
            self.pass_config = config.compilation_config.pass_config
            self.model_dtype = (
                config.model_config.dtype if config.model_config else None
            )
            self.device: str | None = (
                config.device_config.device if config.device_config else None
            )
        else:
            self.compilation_config = None
            self.pass_config = None
            self.model_dtype = None
            self.device = None
        self.pass_name = self.__class__.__name__

    def uuid(self) -> str:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return VllmInductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: str | Any) -> str:
        """
        Utility method to hash the sources of functions or objects.
        :param srcs: strings or objects to add to the hash.
        Objects and functions have their source inspected.
        :return:
        """
        hasher = hashlib.sha256()
        for src in srcs:
            if isinstance(src, str):
                src_str = src
            elif isinstance(src, (types.FunctionType, type)):
                src_str = inspect.getsource(src)
            else:
                # object instance
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(dict_: dict[Any, Any]) -> str:
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.
        :return: A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True

    @staticmethod
    def time_and_log(
        call_fn: Callable[[VllmInductorPass, torch.fx.Graph], None],
    ) -> Callable[[VllmInductorPass, torch.fx.Graph], None]:
        @functools.wraps(call_fn)
        def wrapped(self: VllmInductorPass, graph: torch.fx.Graph) -> None:
            self.begin()
            self.dump_graph(graph, "before")
            call_fn(self, graph)
            self.dump_graph(graph, "after")
            self.end_and_log()

        return wrapped

    def dump_graph(self, graph: torch.fx.Graph, stage: str) -> None:
        i = VllmInductorPass.dump_prefix
        i_str = "" if i is None else f".{i}"
        lazy_format_graph_code(
            f"post_grad{i_str}.{self.pass_name}.{stage}", graph.owning_module
        )

    def begin(self) -> None:
        self._start_time = time.perf_counter_ns()

    def end_and_log(self) -> None:
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)


class CallableInductorPass(VllmInductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(
        self, callable: Callable[[fx.Graph], None], uuid: Any | None = None
    ) -> None:
        super().__init__(config=None)
        self.callable = callable
        self._uuid = self.hash_source(callable) if uuid is None else uuid

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid


def enable_fake_mode(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Applies a FakeTensorMode context. This is useful when you don't want to
    create or run things with real tensors.
    """

    @functools.wraps(fn)
    def fn_new(*args: P.args, **kwargs: P.kwargs) -> R:
        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(*args, **kwargs)

        return result

    return fn_new


class VllmPatternMatcherPass(VllmInductorPass):
    """
    A VllmInductorPass that uses the Inductor pattern matcher.
    Its main use is providing the dump_patterns utility that dumps the
    Inductor pattern matcher patterns into a file, which greatly aids debugging.

    TODO(luka) move more utilities to this pass.
    """

    matched_count: int = 0
    """The number of matched patterns in the pass."""

    _OP_OVERLOAD_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<OpOverload\(op='([^']*)', overload='([^']*)'\)>"
    )

    def _replace_op_overloads(self, string: str) -> str:
        """Replace <OpOverload(..., ...)> with nicer formulations"""
        return str(
            self._OP_OVERLOAD_PATTERN.sub(
                lambda m: f"torch.ops.{m.group(1)}.{m.group(2)}",
                string,
            )
        )

    def dump_patterns(self, config: VllmConfig, pm_pass: PatternMatcherPass) -> None:
        """
        If debug dumping is enabled, dump the Inductor pattern-matcher patterns
        into the debug_dump_path folder next to the dumped fx graphs.

        This method does its best to print something that looks like Python code
        for easier debugging and potentially navigation. If any errors appear in
        the output, please add to this method.

        TODO(luka): use pattern object to manually produce pattern graph
        """
        debug_dump_path = config.compile_debug_dump_path()
        if not debug_dump_path:
            return

        debug_dump_path.mkdir(parents=True, exist_ok=True)

        from vllm.utils.system_utils import unique_filepath

        file_path = unique_filepath(
            lambda i: debug_dump_path / f"patterns.{self.pass_name}.{i}.py"
        )

        with file_path.open("w") as f:
            print(
                f"# This file was produced by VllmPatternMatcherPass."
                f"dump_patterns for {self.pass_name}.\n"
                f"# It does its best to produce valid-Python-looking code but"
                f" please add to dump_patterns if there are any errors.\n\n"
                f"from torch._higher_order_ops.auto_functionalize import "
                f"auto_functionalized as auto_functionalized\n"
                f"from torch._inductor.pattern_matcher import *\n"
                f"vllm = torch.ops.vllm",
                file=f,
            )

            for node, patterns in pm_pass.patterns.items():
                # fix the operator.getitem repr
                if node[1] == operator.getitem:
                    node_repr = f"({repr(node[0])}, operator.getitem)"
                else:
                    node_repr = repr(node)

                node_repr = self._replace_op_overloads(node_repr)

                print(f"\n\n# Patterns for op: {node_repr}", file=f)
                for i, pattern in enumerate(patterns):
                    # reserve auto_functionalized ahead of time
                    pp = PatternPrettyPrinter()
                    pp.namespace.create_name("auto_functionalized", None)

                    # Assemble pattern
                    out_node = pp.pretty_print(pattern.pattern)
                    pattern_repr = "\n".join(
                        [f"def pattern_{i}():"]
                        + [
                            f"{pp.memoized_objs_names[key]} = "
                            f"{pp.memoized_objs_pp[key]}"
                            for key in pp.memoized_objs_names
                        ]
                        + [f"return {out_node}"]
                    ).replace("\n", "\n    ")

                    pattern_repr = self._replace_op_overloads(pattern_repr)
                    print(f"{pattern_repr}\n", file=f)


class PrinterInductorPass(VllmInductorPass):
    def __init__(self, name: str, config: VllmConfig) -> None:
        super().__init__(config)
        self.name = name

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.dump_graph(graph, self.name)
