"""perform_sweep / perform_run with backend="flash" or "auto" (gnm.fast.run_flash).

These check the integration: the same experiments, in the same order and with the same
result structure, as the torch backend; fallback and errors for unsupported
configurations; seeds; and reproducibility. Correctness of the simulation itself is in
test_flash_kernel.py.

    pytest tests/test_flash_sweep.py -v      # CUDA GPU, or TRITON_INTERPRET=1 on CPU
"""

import pytest
import torch

from gnm import BinaryGenerativeParameters
from gnm.defaults import get_binary_network, get_distance_matrix
from gnm.evaluation import ClusteringKS, DegreeKS, EdgeLengthKS
from gnm.fast import adjacency_snapshots_from_edges, flash_available, flash_unsupported_reason
from gnm.fitting import BinarySweepParameters, RunConfig, SweepConfig, perform_run, perform_sweep
from gnm.generative_rules import MatchingIndex, Neighbours

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not flash_available(DEVICE),
    reason="needs Triton and a CUDA GPU (or TRITON_INTERPRET=1 set before importing triton)",
)

STEPS = 30  # small: these tests also run in Triton's CPU interpreter


def sweep_config(num_simulations=3, preferential=("powerlaw",), rules=None, **kw):
    params = BinarySweepParameters(
        eta=torch.tensor([-3.0, -1.0]),
        gamma=torch.tensor([0.2, 0.5]),
        lambdah=torch.tensor([0.0]),
        distance_relationship_type=["powerlaw"],
        preferential_relationship_type=list(preferential),
        heterochronicity_relationship_type=["powerlaw"],
        generative_rule=rules or [MatchingIndex()],
        num_iterations=[STEPS],
        **kw,
    )
    return SweepConfig(
        binary_sweep_parameters=params,
        num_simulations=num_simulations,
        distance_matrix=[get_distance_matrix()],
    )


def evaluations():
    return [DegreeKS(), ClusteringKS(), EdgeLengthKS(get_distance_matrix())]


def sweep(config, backend, **kw):
    return perform_sweep(
        config,
        binary_evaluations=evaluations(),
        real_binary_matrices=get_binary_network(),
        device=DEVICE,
        backend=backend,
        **kw,
    )


def test_same_structure_as_torch_backend():
    config = sweep_config()
    ref = sweep(config, "torch")
    new = sweep(config, "flash")
    assert len(new) == len(ref) == 4
    for r, f in zip(ref, new):
        rb, fb = r.run_config.binary_parameters, f.run_config.binary_parameters
        assert (float(rb.eta), float(rb.gamma)) == (float(fb.eta), float(fb.gamma))
        assert r.evaluation_results.binary_evaluations.keys() == f.evaluation_results.binary_evaluations.keys()
        for k, v in r.evaluation_results.binary_evaluations.items():
            assert f.evaluation_results.binary_evaluations[k].shape == v.shape
            assert f.evaluation_results.binary_evaluations[k].device.type == "cpu"
        assert f.model.adjacency_matrix.shape == r.model.adjacency_matrix.shape
        assert f.model.adjacency_matrix.dtype == r.model.adjacency_matrix.dtype
        assert (f.model.adjacency_matrix.sum((-1, -2)) == 2 * STEPS).all()
        assert f.run_history.added_edges.shape == r.run_history.added_edges.shape
        assert f.run_history.adjacency_snapshots is None  # see run_flash
        assert f.clamp_count == 0


def test_snapshots_from_edges():
    (e,) = sweep(sweep_config(), "flash")[:1]
    snaps = adjacency_snapshots_from_edges(e.run_history.added_edges, 90)
    assert snaps.shape == (STEPS, 3, 90, 90)
    assert torch.equal(snaps[-1], e.model.adjacency_matrix)
    assert torch.equal(snaps.sum((-1, -2))[:, 0], 2 * torch.arange(1, STEPS + 1, dtype=torch.float32))


def test_reproducible_with_manual_seed():
    torch.manual_seed(123)
    a = sweep(sweep_config(), "flash")
    torch.manual_seed(123)
    b = sweep(sweep_config(), "flash")
    for x, y in zip(a, b):
        assert torch.equal(x.run_history.added_edges, y.run_history.added_edges)


def test_small_batches_split_configurations():
    """Configurations split across launches are reassembled in order."""
    from gnm.fast import run_flash

    configs = list(sweep_config(num_simulations=5))
    exps, times = run_flash(configs, device=DEVICE, max_batch_size=3)
    assert len(exps) == 4 and len(times) == 4
    for rc, e in zip(configs, exps):
        assert e.run_config is rc
        assert e.model.adjacency_matrix.shape == (5, 90, 90)
        assert (e.model.adjacency_matrix.sum((-1, -2)) == 2 * STEPS).all()


def test_seed_networks_and_updates_per_iteration():
    seed = get_binary_network()[0].clone()
    keep = torch.rand(seed.shape, generator=torch.Generator().manual_seed(0)) < 0.25
    seed = seed * (keep | keep.T)  # a sparse subset of the target, symmetric
    config = sweep_config(binary_updates_per_iteration=[2])
    config.seed_adjacency_matrix = [seed]
    for e in sweep(config, "flash"):
        A = e.model.adjacency_matrix
        assert (A >= seed).all(), "seed edges kept"
        assert (A.sum((-1, -2)) == seed.sum() + 2 * 2 * STEPS).all()
        assert e.run_history.added_edges.shape[0] == 2 * STEPS


def test_auto_falls_back_and_flash_raises():
    config = sweep_config(preferential=("powerlaw", "exponential"))
    reasons = [flash_unsupported_reason(rc) for rc in config]
    assert sum(r is None for r in reasons) == 4 and sum(r is not None for r in reasons) == 4

    with pytest.raises(ValueError, match="not supported by backend='flash'"):
        sweep(config, "flash")

    exps = sweep(config, "auto")
    assert len(exps) == 8
    for rc, e in zip(config, exps):
        flash = flash_unsupported_reason(rc) is None
        assert (e.run_history.adjacency_snapshots is None) == flash  # torch keeps snapshots


@pytest.mark.parametrize(
    "change,expected",
    [
        (dict(generative_rule=Neighbours()), "generative rule"),
        (dict(generative_rule=MatchingIndex(divisor="union")), "divisor"),
        (dict(distance_relationship_type="exponential"), "distance relationship"),
        (dict(heterochronicity_relationship_type="exponential", lambdah=1.0), "heterochronicity"),
    ],
)
def test_unsupported_reasons(change, expected):
    base = dict(
        eta=-2.0, gamma=0.3, lambdah=0.0, distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(), num_iterations=10,
    )
    base.update(change)
    rc = RunConfig(binary_parameters=BinaryGenerativeParameters(**base), num_simulations=1,
                   distance_matrix=get_distance_matrix())
    assert expected in flash_unsupported_reason(rc)


def test_supported_lambdah_without_heterochronous_matrix():
    rc = RunConfig(
        binary_parameters=BinaryGenerativeParameters(
            eta=-2.0, gamma=0.3, lambdah=2.0, distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
            generative_rule=MatchingIndex(), num_iterations=10),
        num_simulations=2, distance_matrix=get_distance_matrix())
    assert flash_unsupported_reason(rc) is None
    e = perform_run(rc, binary_evaluations=evaluations(), real_binary_matrices=get_binary_network(),
                    device=DEVICE, backend="flash")
    assert e.model.adjacency_matrix.sum() == 2 * 2 * 10
    assert set(e.evaluation_results.binary_evaluations) == {str(c) for c in evaluations()}


def test_bayesian_with_flash_raises():
    with pytest.raises(NotImplementedError):
        sweep(sweep_config(), "flash", method="bayesian")
