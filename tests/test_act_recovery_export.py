"""Exported expert RGB observations must remain pixel- and time-identical."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from export_act_recovery_dataset import encode_lossless,verify_video


class RecoveryExportTests(unittest.TestCase):
    def test_lossless_camera_video_preserves_every_pixel_and_timestamp(self):
        frames=np.random.default_rng(123).integers(0,256,(4,256,256,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'camera.mp4'
            encode_lossless(frames,path)
            self.assertEqual(verify_video(path,frames),4)


if __name__=='__main__':unittest.main()
