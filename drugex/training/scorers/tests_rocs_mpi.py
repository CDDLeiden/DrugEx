"""TDD: OpenEye ROCS `-mpi_np` single-node parallelism flag (throughput fix for
the cross-dataset enrichment benchmark — job 10416 timed out at 3600s, single
core). Run: python -m drugex.training.scorers.tests_rocs_mpi
"""
import unittest


class TestRocsMpi(unittest.TestCase):
    def _scorer(self, mpi_np):
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        s = OpenEyeROCSScorer.__new__(OpenEyeROCSScorer)
        s.binary_path = "rocs"
        s.mpi_np = mpi_np
        s.score_type = "TanimotoCombo"
        s.color_force_field = "ImplicitMillsDean"
        s.optimize = True
        s.shape_only = False
        s.color_optimize = True
        return s

    def test_mpi_np_flag_emitted_when_gt_one(self):
        cmd = self._scorer(44)._build_rocs_command("q.sdf", "db.sdf", "out/r")
        self.assertIn("-mpi_np", cmd)
        self.assertIn("44", cmd)
        # -mpi_np must come right after the binary (OpenEye CLI convention)
        i = cmd.index("-mpi_np")
        self.assertEqual(cmd[0], "rocs")
        self.assertEqual(cmd[i + 1], "44")

    def test_mpi_np_absent_when_one(self):
        cmd = self._scorer(1)._build_rocs_command("q.sdf", "db.sdf", "out/r")
        self.assertNotIn("-mpi_np", cmd)


if __name__ == "__main__":
    unittest.main()
