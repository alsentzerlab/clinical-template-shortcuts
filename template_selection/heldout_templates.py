"""The ten held-out templates (H1-H10) used in the paper, ordered by readmission rate.

sample_heldout_templates.py drew twelve candidates by stratified random sampling;
the two lowest-rate draws (398647: 2.2%, 5 readmissions; 478352: 4.6%, 11
readmissions) were not used. The sampler with its current defaults does not
regenerate this draw, so this list -- not a rerun -- defines the held-out set.

Columns: label, template_id, notes, 30-day readmission rate.
"""

HELDOUT = [
    ("H1",   899007, 125, 0.104),
    ("H2",   714564, 292, 0.147),
    ("H3",   593537, 117, 0.188),
    ("H4",  1327338, 140, 0.207),
    ("H5",  1459783, 159, 0.245),
    ("H6",   181781, 774, 0.307),
    ("H7",   803224, 189, 0.323),
    ("H8",  1137807, 835, 0.326),
    ("H9",   529413, 160, 0.356),
    ("H10",  966314, 186, 0.457),
]

HELDOUT_IDS = [t for _, t, _, _ in HELDOUT]
HELDOUT_LABEL = {t: h for h, t, _, _ in HELDOUT}
