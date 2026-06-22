"""Unit tests for the Experiment B (prior-guided BO) building blocks.

Covers:

- :class:`benchmarks.oriented.OrientedBenchmark` raw-scale orientation;
- :class:`common.priors.GaussianPrior` evaluation and unit-cube transform;
- :class:`common.priors.PriorMeanModule` ``f_prior = baseline + amplitude *
  pi`` on the y-scale, lambda training / freezing and state-dict
  serialization;
- :class:`common.priors.PriorGPModelBuilder` snapshot reconstruction;
- the three acquisitions of :mod:`acquisitions.prior_guided`;
- raw / processed dataclass JSON roundtrips.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from acquisitions.prior_guided import AlphaPiUCB, LambdaPriorUCB, VanillaUCB
from benchmarks.oriented import (
    KNOWN_OUTPUT_RANGES,
    OrientedBenchmark,
    build_oriented_benchmark,
)
from common.models import (
    GPModelConfig,
    load_state_dict_into,
    serialize_state_dict,
)
from common.priors import GaussianPrior, PriorGPModelBuilder, PriorMeanModule
from data.raw_prior import (
    PriorMethodTrajectory,
    PriorSeedMetadata,
    PriorStepRecord,
    RawPriorSeedData,
)
from data.raw import InitialDesign, ModelSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def forrester() -> OrientedBenchmark:
    return build_oriented_benchmark(
        {"benchmark": {"name": "forrester_1d"}, "noise_std": 0.5}
    )


@pytest.fixture()
def branin() -> OrientedBenchmark:
    return build_oriented_benchmark(
        {"benchmark": {"name": "branin_2d"}, "noise_std": 2.0}
    )


def _fit_data(problem, n: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    X = problem.test_set(n=n, seed=seed)
    obs = problem.observe(X)
    return X, obs.y_obs


# ---------------------------------------------------------------------------
# OrientedBenchmark
# ---------------------------------------------------------------------------

class TestOrientedBenchmark:
    def test_minimization_maps_optimum_to_f_opt(self, forrester):
        # g = -f_raw; the inner global minimizer maps to the oriented optimum.
        assert forrester.f_opt == pytest.approx(6.020740055735769, abs=1e-9)
        x_star = torch.tensor([[0.7572491]], dtype=torch.double)
        g = forrester.f(x_star)
        assert g.shape == (1, 1)
        assert g.item() == pytest.approx(forrester.f_opt, abs=1e-3)

    def test_objective_is_raw_scale_not_unit_interval(self, branin):
        # The objective keeps Branin's raw scale (oriented to maximization):
        # g in [-f_max, -f_min] = [-308.13, -0.398], never squashed to [0, 1].
        X = branin.test_set(n=512, seed=3)
        g = branin.f(X)
        assert float(g.min()) >= branin.f_lower - 1e-6
        assert float(g.max()) <= branin.f_upper + 1e-6
        # Definitely not normalized: every value is well below 0.
        assert float(g.max()) < 0.0

    def test_observe_adds_noise_on_raw_scale(self, forrester):
        torch.manual_seed(0)
        X = torch.full((512, 1), 0.3, dtype=torch.double)
        obs = forrester.observe(X)
        residual = obs.y_obs - obs.f_noiseless
        assert float(residual.std()) == pytest.approx(0.5, rel=0.2)

    def test_known_ranges_cover_exp_b_benchmarks(self):
        for name in ("forrester_1d", "branin_2d", "hartmann_3d", "hartmann_6d"):
            assert name in KNOWN_OUTPUT_RANGES

    def test_hartmann_3d_oriented_optimum(self):
        p = build_oriented_benchmark({"benchmark": {"name": "hartmann_3d"}})
        assert p.f_opt == pytest.approx(3.86278214782076, abs=1e-9)
        x_star = torch.tensor([[0.114614, 0.555649, 0.852547]], dtype=torch.double)
        assert p.f(x_star).item() == pytest.approx(p.f_opt, abs=1e-4)

    def test_prior_mean_scale_spans_oriented_range(self, forrester):
        baseline, amplitude = forrester.prior_mean_scale()
        assert baseline == pytest.approx(forrester.f_lower)
        assert amplitude == pytest.approx(forrester.f_upper - forrester.f_lower)
        # f_prior peaks at the optimum value when pi = 1.
        assert baseline + amplitude == pytest.approx(forrester.f_opt)

    def test_rejects_inner_noise_std(self):
        with pytest.raises(ValueError, match="top level"):
            build_oriented_benchmark(
                {"benchmark": {"name": "forrester_1d", "noise_std": 0.5}}
            )

    def test_metadata_contains_orientation(self, forrester):
        meta = forrester.metadata()
        assert meta["orientation"]["base_benchmark"] == "forrester_1d"
        assert meta["orientation"]["direction"] == "min"
        assert meta["orientation"]["f_opt"] == pytest.approx(forrester.f_opt)

    def test_name_has_no_normalized_suffix(self, forrester):
        assert forrester.name == "forrester_1d"


# ---------------------------------------------------------------------------
# DoubleGaussianWell1D (oriented)
# ---------------------------------------------------------------------------

class TestDoubleGaussianWell1D:
    def _oriented(self, **bench_kwargs):
        return build_oriented_benchmark(
            {"benchmark": {"name": "double_gaussian_well_1d", **bench_kwargs}}
        )

    def test_two_peaks_with_correct_heights(self):
        p = self._oriented()
        x = torch.linspace(0, 1, 2001, dtype=torch.double).unsqueeze(-1)
        g = p.f(x).squeeze(-1)
        # Global oriented peak ~1.0 at the deep well (x = 0.70).
        i_max = int(g.argmax())
        assert x[i_max].item() == pytest.approx(0.70, abs=5e-3)
        assert g[i_max].item() == pytest.approx(p.f_opt, abs=1e-4)
        assert g[i_max].item() == pytest.approx(1.0, abs=1e-2)
        # Deceptive local oriented peak at the shallow well (x = 0.25) ~ 0.6.
        left = x.squeeze(-1) < 0.5
        i_loc = int(g[left].argmax())
        assert x[left][i_loc].item() == pytest.approx(0.25, abs=5e-3)
        assert g[left][i_loc].item() == pytest.approx(0.6, abs=0.01)

    def test_output_range_used_by_wrapper(self):
        p = self._oriented(deep_depth=2.0, shallow_depth=0.5)
        f_min, f_max, direction = p.inner.output_range()
        assert direction == "min"
        assert f_min == pytest.approx(-2.0, abs=1e-2)
        # The wrapper stores the RAW inner extrema; f_opt = -f_min.
        assert p.f_min == f_min and p.f_max == f_max
        assert p.f_opt == pytest.approx(-f_min, abs=1e-9)

    def test_shallow_must_be_shallower(self):
        with pytest.raises(ValueError, match="shallow_depth"):
            self._oriented(deep_depth=0.5, shallow_depth=0.8)

    def test_centers_must_lie_in_domain(self):
        with pytest.raises(ValueError, match="inside the domain"):
            self._oriented(deep_center=1.5)

    def test_from_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            self._oriented(depth_typo=1.0)


# ---------------------------------------------------------------------------
# GaussianPrior
# ---------------------------------------------------------------------------

class TestGaussianPrior:
    def test_value_at_center_is_one(self):
        prior = GaussianPrior(center=[0.3, 0.7], lengthscale=0.2)
        x = torch.tensor([[0.3, 0.7]], dtype=torch.double)
        assert prior.evaluate(x).item() == pytest.approx(1.0)

    def test_isotropic_formula(self):
        l = 0.25
        prior = GaussianPrior(center=[0.0], lengthscale=l)
        x = torch.tensor([[0.5]], dtype=torch.double)
        expected = math.exp(-0.5 * (0.5 / l) ** 2)
        assert prior.evaluate(x).item() == pytest.approx(expected)

    def test_unit_cube_transform_is_exact(self):
        bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double)
        prior = GaussianPrior(center=[3.14159, 2.275], lengthscale=2.25)
        unit_prior = prior.transformed_to_unit_cube(bounds)
        X = torch.tensor([[0.0, 5.0], [-2.0, 12.0], [9.0, 1.0]], dtype=torch.double)
        U = (X - bounds[0]) / (bounds[1] - bounds[0])
        assert torch.allclose(prior.evaluate(X), unit_prior.evaluate(U))

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            GaussianPrior.from_config(
                {"center": [0.5], "lengthscale": 0.1}, dim=2
            )

    def test_nonpositive_lengthscale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GaussianPrior(center=[0.5], lengthscale=0.0)

    def test_roundtrip_dict(self):
        prior = GaussianPrior(center=[0.1, 0.9], lengthscale=[0.2, 0.3])
        again = GaussianPrior.from_config(prior.to_dict(), dim=2)
        assert torch.allclose(prior.center, again.center)
        assert torch.allclose(prior.lengthscale, again.lengthscale)


# ---------------------------------------------------------------------------
# PriorMeanModule + PriorGPModelBuilder
# ---------------------------------------------------------------------------

class TestPriorMeanGP:
    def _builder(self, problem, prior, learn_lambda):
        cfg = GPModelConfig(
            noise_std=0.01, normalize_input=True, standardize_output=False
        )
        baseline, amplitude = problem.prior_mean_scale()
        return PriorGPModelBuilder(
            cfg, bounds=problem.bounds, prior=prior, learn_lambda=learn_lambda,
            prior_baseline=baseline, prior_amplitude=amplitude,
        )

    def test_mean_module_default_is_lambda_times_pi(self):
        # Default (baseline, amplitude) = (0, 1) reduces to lambda * pi.
        prior = GaussianPrior(center=[0.4], lengthscale=0.2)
        mean = PriorMeanModule(prior, lambda_init=0.7, learn_lambda=True)
        x = torch.tensor([[0.4], [0.8]], dtype=torch.double)
        assert torch.allclose(mean(x), 0.7 * prior.evaluate(x))

    def test_mean_module_uses_baseline_and_amplitude(self):
        # f_prior = baseline + amplitude * pi, scaled by lambda.
        prior = GaussianPrior(center=[0.4], lengthscale=0.2)
        mean = PriorMeanModule(
            prior, lambda_init=2.0, learn_lambda=False, baseline=-3.0, amplitude=10.0
        )
        x = torch.tensor([[0.4], [0.8]], dtype=torch.double)
        expected = 2.0 * (-3.0 + 10.0 * prior.evaluate(x))
        assert torch.allclose(mean(x), expected)

    def test_fixed_lambda_stays_at_one_after_fit(self, forrester):
        prior = GaussianPrior(center=[0.757249], lengthscale=0.15)
        X, y = _fit_data(forrester)
        model = self._builder(forrester, prior, learn_lambda=False).fit(X, y)
        assert isinstance(model.mean_module, PriorMeanModule)
        assert model.mean_module.lambda_value == pytest.approx(1.0)

    def test_adaptive_lambda_moves_after_fit(self, forrester):
        prior = GaussianPrior(center=[0.757249], lengthscale=0.15)
        X, y = _fit_data(forrester)
        model = self._builder(forrester, prior, learn_lambda=True).fit(X, y)
        assert model.mean_module.lambda_value != pytest.approx(1.0)

    def test_fixed_and_adaptive_share_state_dict_keys(self, forrester):
        prior = GaussianPrior(center=[0.757249], lengthscale=0.15)
        X, y = _fit_data(forrester)
        m_fixed = self._builder(forrester, prior, learn_lambda=False).construct(X, y)
        m_adapt = self._builder(forrester, prior, learn_lambda=True).construct(X, y)
        assert set(m_fixed.state_dict()) == set(m_adapt.state_dict())
        assert any("raw_lambda" in k for k in m_fixed.state_dict())

    def test_snapshot_reconstruction_matches_fitted_posterior(self, branin):
        prior = GaussianPrior(center=[3.14159, 2.275], lengthscale=2.25)
        X, y = _fit_data(branin, n=16, seed=2)
        builder = self._builder(branin, prior, learn_lambda=True)
        model = builder.fit(X, y)
        serial = serialize_state_dict(model)

        rebuilt = builder.construct(X, y)
        load_state_dict_into(rebuilt, serial)
        rebuilt.eval()

        Xq = branin.test_set(n=32, seed=9)
        with torch.no_grad():
            p1, p2 = model.posterior(Xq), rebuilt.posterior(Xq)
        assert torch.allclose(p1.mean, p2.mean)
        assert torch.allclose(p1.variance, p2.variance)

    def test_prior_with_standardize_output_raises(self, forrester):
        prior = GaussianPrior(center=[0.5], lengthscale=0.2)
        cfg = GPModelConfig(
            noise_std=0.01, normalize_input=True, standardize_output=True
        )
        with pytest.raises(ValueError, match="standardize_output"):
            PriorGPModelBuilder(cfg, bounds=forrester.bounds, prior=prior)

    def test_extract_hyperparameters_includes_lambda(self, forrester):
        prior = GaussianPrior(center=[0.757249], lengthscale=0.15)
        X, y = _fit_data(forrester)
        builder = self._builder(forrester, prior, learn_lambda=True)
        params = builder.extract_hyperparameters(builder.fit(X, y))
        assert "lambda" in params and params["lambda_learned"] is True


# ---------------------------------------------------------------------------
# Acquisitions
# ---------------------------------------------------------------------------

class TestAcquisitions:
    @pytest.fixture()
    def model_and_prior(self, forrester):
        prior = GaussianPrior(center=[0.757249], lengthscale=0.15)
        X, y = _fit_data(forrester, n=8)
        cfg = GPModelConfig(
            noise_std=0.01, normalize_input=True, standardize_output=True
        )
        model = PriorGPModelBuilder(cfg, bounds=forrester.bounds).fit(X, y)
        return model, prior

    def test_alpha_zero_equals_vanilla(self, model_and_prior):
        model, prior = model_and_prior
        X = torch.rand(16, 1, 1, dtype=torch.double)
        a0 = AlphaPiUCB(model, prior=prior, alpha=0.0, beta=2.0)
        v = VanillaUCB(model, beta=2.0)
        assert torch.allclose(a0(X), v(X))

    def test_alpha_pi_weighting_formula(self, model_and_prior):
        """a(x) = mu + [w(x) / (1-a)] * sqrt(beta) * sigma,  w = (1-a) + a*pi."""
        model, prior = model_and_prior
        alpha, beta = 0.6, 2.0
        X = torch.rand(8, 1, 1, dtype=torch.double)
        acq = AlphaPiUCB(model, prior=prior, alpha=alpha, beta=beta)
        with torch.no_grad():
            posterior = model.posterior(X)
            mu = posterior.mean.squeeze(-1).squeeze(-1)
            sigma = posterior.variance.clamp_min(1e-10).sqrt().squeeze(-1).squeeze(-1)
        w = (1 - alpha) + alpha * prior.evaluate(X.squeeze(-2))
        expected = mu + (w / (1 - alpha)) * math.sqrt(beta) * sigma
        assert torch.allclose(acq(X), expected)

    def test_alpha_pi_dominates_vanilla_pointwise(self, model_and_prior):
        """The normalized weight is >= 1, so a(x) >= vanilla UCB everywhere."""
        model, prior = model_and_prior
        X = torch.linspace(0, 1, 256, dtype=torch.double).reshape(-1, 1, 1)
        for alpha in (0.25, 0.5, 0.9):
            a_pi = AlphaPiUCB(model, prior=prior, alpha=alpha, beta=2.0)(X)
            a_v = VanillaUCB(model, beta=2.0)(X)
            assert torch.all(a_pi >= a_v - 1e-12)
        # alpha = 1/2: multiplier 1 + pi in [1, 2] -> bonus at the prior
        # center is exactly twice the vanilla bonus.
        c = prior.center.reshape(1, 1, -1)
        a_pi_c = AlphaPiUCB(model, prior=prior, alpha=0.5, beta=2.0)(c)
        with torch.no_grad():
            post = model.posterior(c.squeeze(1))
            mu = post.mean.squeeze()
            sigma = post.variance.clamp_min(1e-10).sqrt().squeeze()
        assert torch.allclose(a_pi_c.squeeze(), mu + 2.0 * math.sqrt(2.0) * sigma)

    def test_lambda_prior_ucb_is_plain_ucb_on_posterior(self, model_and_prior):
        model, _ = model_and_prior
        X = torch.rand(8, 1, 1, dtype=torch.double)
        assert torch.allclose(
            LambdaPriorUCB(model, beta=2.0)(X), VanillaUCB(model, beta=2.0)(X)
        )

    def test_q_greater_than_one_raises(self, model_and_prior):
        model, prior = model_and_prior
        X = torch.rand(3, 2, 1, dtype=torch.double)
        for acq in (
            VanillaUCB(model),
            AlphaPiUCB(model, prior=prior),
            LambdaPriorUCB(model),
        ):
            with pytest.raises(ValueError, match="q=1"):
                acq(X)

    def test_invalid_alpha_raises(self, model_and_prior):
        model, prior = model_and_prior
        with pytest.raises(ValueError, match="alpha"):
            AlphaPiUCB(model, prior=prior, alpha=1.5)
        # alpha = 1 is rejected: the 1 / (1 - alpha) normalization diverges.
        with pytest.raises(ValueError, match="alpha"):
            AlphaPiUCB(model, prior=prior, alpha=1.0)


# ---------------------------------------------------------------------------
# Raw dataclass roundtrips
# ---------------------------------------------------------------------------

class TestRawPriorSchemas:
    def test_raw_seed_roundtrip(self):
        step = PriorStepRecord(
            iteration=1, x=[0.5], y_obs=5.9, f_noiseless=5.91,
            acq_value=6.2, train_size=3,
            posterior_mean_at_candidate=5.88, posterior_std_at_candidate=0.05,
            prior_value_at_candidate=0.7, lambda_value=0.4,
        )
        snapshot = ModelSnapshot(
            iteration=1, model_type="FixedNoiseGP", kernel_name="matern52",
            noise_std=0.5, state_dict={"mean_module.raw_lambda": 0.4},
            hyperparameters={"lambda": 0.4},
        )
        method = PriorMethodTrajectory(
            method_name="lambda_pi_adaptive_ucb",
            train_label="y_obs",
            model_kind="prior_mean_gp",
            learn_lambda=True,
            trajectory=[step],
            model_snapshots=[snapshot],
        )
        raw = RawPriorSeedData(
            metadata=PriorSeedMetadata(
                experiment="exp_b_prior_guided",
                problem_type="synthetic",
                benchmark="forrester_1d",
                base_benchmark="forrester_1d",
                seed=0, budget=1, initial_design_size=3,
                noise_std=0.5, dim=1,
                bounds=[[0.0], [1.0]],
                orientation={
                    "f_min": -6.0, "f_max": 15.8, "direction": "min",
                    "f_lower": -15.8, "f_upper": 6.0, "f_opt": 6.0,
                },
                prior={"center": [0.757], "lengthscale": [0.15], "label": "good"},
                f_opt=6.0, alpha=0.5, beta=2.0,
            ),
            initial_design=InitialDesign(
                X=[[0.1], [0.5], [0.9]],
                y_obs=[2.0, 5.8, 1.6],
                f_noiseless=[2.1, 5.79, 1.61],
            ),
            methods={"lambda_pi_adaptive_ucb": method},
        )
        again = RawPriorSeedData.from_dict(raw.to_dict())
        assert again.to_dict() == raw.to_dict()
        assert again.metadata.f_opt == pytest.approx(6.0)
        assert again.metadata.orientation["direction"] == "min"
        m = again.methods["lambda_pi_adaptive_ucb"]
        assert m.learn_lambda is True
        assert m.trajectory[0].lambda_value == pytest.approx(0.4)
