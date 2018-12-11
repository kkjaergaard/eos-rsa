#!/usr/bin/env python3

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

src_dir = Path(__file__).parent.joinpath("../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.series import RSA_Scene_Segment_Motion
from rsa.dataset import RSA_Dataset
from rsa.series import RSA_Series_Logic


@click.command()
@click.option("--from", "_from", required=True, nargs=1, type=str)
@click.option("--to", required=True, nargs=1, type=str)
@click.option("--reference-segment", required=True, nargs=1, type=str)
@click.argument("filename", nargs=1, type=click.Path(exists=True))
@click.argument("segments", nargs=-1, type=str)
def find_motion(_from, to, reference_segment, filename, segments):
    rsa_series = RSA_Series_Logic(
        RSA_Dataset(filename)
    )

    # find reference motion
    reference_motion = RSA_Scene_Segment_Motion()
    reference_motion._from = rsa_series.assessment(_from)
    reference_motion.to = rsa_series.assessment(to)
    reference_motion.segment_name = reference_segment

    R_ref, t_ref, _ = reference_motion.find_motion()
    T_ref = np.identity(4)
    T_ref[:3, :3] = R_ref
    T_ref[:3, 3] = t_ref

    res = []

    for scene_segment_name in rsa_series.scene_segment_list():
        # skip the reference segment
        if scene_segment_name == reference_segment:
            continue

        # skip segments not specified on the terminal
        if len(segments) > 0:
            if not scene_segment_name in segments:
                continue

        # find reference motion
        motion = RSA_Scene_Segment_Motion()
        motion._from = rsa_series.assessment(_from)
        motion.to = rsa_series.assessment(to)
        motion.segment_name = scene_segment_name

        R, t, dists = motion.find_motion(local=True, T_ref=T_ref)

        res.append({
            "segment": scene_segment_name,
            "t": t.tolist(),
            "R": R.tolist(),
            "ME": np.mean(dists),
            "MSE": np.mean(np.square(dists)),
            "RMSE": np.sqrt(np.mean(np.square(dists))),
            "status": "ok"
        })

    click.echo(json.dumps(res, indent=4))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    find_motion()
