"""
Unit Tests for TensorRT Integration

Tests the TensorRT integration components using mocks.

Author: Claude Code
Date: 2025-12-26
Phase: 1-B
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTensorRTIntegration(unittest.TestCase):
    """Test TensorRT integration components."""

    def setUp(self):
        """Set up mocks."""
        # Mock all dependencies
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()
        sys.modules['tensorrt'] = MagicMock()
        sys.modules['pycuda'] = MagicMock()
        sys.modules['pycuda.driver'] = MagicMock()
        sys.modules['pycuda.autoinit'] = MagicMock()

    def test_module_imports(self):
        """Test that TensorRT modules can be imported."""
        try:
            from sam3_video import trt_engine
            from sam3_video import trt_session_manager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import TensorRT modules: {e}")

    def test_version_upgrade(self):
        """Test that version was upgraded to 0.2.0."""
        import sam3_video
        self.assertEqual(sam3_video.__version__, '0.2.0')

    def test_new_exports(self):
        """Test that new classes are exported."""
        import sam3_video

        exports = sam3_video.__all__
        self.assertIn('Sam3VideoSessionManagerTRT', exports)
        self.assertIn('TensorRTEngine', exports)
        self.assertIn('TensorRTEngineManager', exports)
        self.assertIn('MemoryBankConfig', exports)

    def test_create_session_manager_function(self):
        """Test create_session_manager helper function."""
        from sam3_video import create_session_manager

        # Should not raise
        session = create_session_manager(use_tensorrt=False)
        self.assertIsNotNone(session)

    def test_get_version_info(self):
        """Test get_version_info function."""
        from sam3_video import get_version_info

        info = get_version_info()

        self.assertIn('version', info)
        self.assertIn('tensorrt_support', info)
        self.assertIn('phase', info)

        self.assertEqual(info['version'], '0.2.0')


class TestFileStructure(unittest.TestCase):
    """Test that all Phase 1-B files exist."""

    def test_trt_files_exist(self):
        """Test that TensorRT files exist."""
        base_path = Path(__file__).parent.parent

        required_files = [
            'sam3_video/trt_engine.py',
            'sam3_video/trt_session_manager.py',
            'examples/video_tracking_demo_trt.py',
            'benchmarks/video_benchmark_trt.py',
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            self.assertTrue(
                full_path.exists(),
                f"Required TensorRT file not found: {file_path}"
            )


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTensorRTIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStructure))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("SAM3 TensorRT Integration - Unit Tests")
    print("=" * 70)
    print()

    result = run_tests()

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
