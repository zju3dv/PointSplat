from typing import Any

from ...utils.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_bounded_v2 import ViewSamplerBoundedV2, ViewSamplerBoundedV2Cfg
from .view_sampler_fps import ViewSamplerFPS, ViewSamplerFPSCfg
from .view_sampler_uniform import ViewSamplerUniform, ViewSamplerUniformCfg


VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "boundedv2": ViewSamplerBoundedV2,
    "fps": ViewSamplerFPS,
    "uniform": ViewSamplerUniform,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | ViewSamplerBoundedV2Cfg
    | ViewSamplerFPSCfg
    | ViewSamplerUniformCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
