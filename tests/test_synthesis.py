"""Tests for anti-overfitting synthesis engines."""

import pytest

from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.synthesis.cascade_guard import CascadeGuard
from polyharness.synthesis.compiler import DatasetCompiler
from polyharness.synthesis.observation import ObservationTransformer
from polyharness.synthesis.perturbation import SyntaxPerturber


@pytest.fixture
def sample_trajectory():
    return get_standard_benchmarks()[0]


def test_syntax_perturber(sample_trajectory):
    perturber = SyntaxPerturber(seed=42)
    p_traj = perturber.perturb_trajectory(sample_trajectory, parameter_casing_prob=1.0)

    assert p_traj.id != sample_trajectory.id
    # Check that tool parameters or step call arguments were perturbed
    tool = p_traj.tools[0]
    param_names = list(tool.parameters.keys())
    # max_price in snake_case should be converted to maxPrice in camelCase
    assert "maxPrice" in param_names or "query" in param_names


def test_observation_transformer(sample_trajectory):
    transformer = ObservationTransformer()

    # Test html to markdown
    html = "<div class='content'><h1>Header</h1><p>Description text</p><button id='btn1'>Submit</button></div>"
    md = transformer.html_to_markdown(html)
    assert "# Header" in md
    assert "Description text" in md
    assert "[Button:" in md

    # Test html to accessibility tree
    axtree = transformer.html_to_accessibility_tree(html)
    assert "button" in axtree
    assert "heading" in axtree

    # Test DOM compaction
    noisy_html = "<div class='container' style='color: red'><script>console.log('noise');</script><svg><path d='M0'/></svg><button id='btn-buy' onclick='alert()'>Buy</button></div>"
    compacted = transformer.compact_dom(noisy_html)
    assert "<script>" not in compacted
    assert "<svg>" not in compacted
    assert "btn-buy" in compacted
    assert "style=" not in compacted


def test_cascade_guard(sample_trajectory):
    guard = CascadeGuard(seed=42)
    initial_steps = len(sample_trajectory.steps)
    guarded = guard.inject_recovery_turn(sample_trajectory)

    # Injected a failing step + tagged a recovery turn
    assert len(guarded.steps) == initial_steps + 1
    assert any(s.is_recovery_turn for s in guarded.steps)
    assert any(any(r.is_error for r in s.tool_results) for s in guarded.steps)


def test_dataset_compiler(sample_trajectory, tmp_path):
    compiler = DatasetCompiler(seed=42)

    # Test single harness export
    single_records = compiler.compile_single_harness([sample_trajectory], target_harness="openai")
    assert len(single_records) == 1
    assert "messages" in single_records[0]

    # Test multi-harness mixture export
    mixture_records = compiler.compile_multi_harness_mixture([sample_trajectory])
    assert len(mixture_records) == 1
    assert "_polyharness_meta" in mixture_records[0]

    # Test export to file
    out_file = tmp_path / "train.jsonl"
    compiler.export_to_file(mixture_records, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
