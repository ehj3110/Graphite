"""
1 mm³ cube case study — Split-P piecewise / linear graded vs cross-hatch woodpile.

Outputs: ``outputs/case_studies/cube_1mm/``
Docs: ``docs/CASE_STUDY_CUBE_1MM.md``
"""

from graphite.case_studies.cube_1mm.specs import (
    default_output_dir,
    repo_root,
    splitp_linear_graded_stem,
    splitp_piecewise_stem,
    woodpile_stem_default,
    woodpile_stem_splitp_match,
)

__all__ = [
    "default_output_dir",
    "repo_root",
    "splitp_linear_graded_stem",
    "splitp_piecewise_stem",
    "woodpile_stem_default",
    "woodpile_stem_splitp_match",
]
