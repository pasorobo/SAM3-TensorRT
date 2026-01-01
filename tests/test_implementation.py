"""
Unit Tests for SAM3 Video Tracking

Tests the core functionality of the video tracking implementation
using mocks to avoid heavy dependencies.

Author: Claude Code
Date: 2025-12-26
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from collections import deque

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryBank(unittest.TestCase):
    """Test MemoryBank functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock all dependencies before import
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()

        # Mock numpy
        self.numpy_mock = sys.modules['numpy']
        self.numpy_mock.ndarray = Mock
        self.numpy_mock.zeros = Mock(return_value=Mock())
        self.numpy_mock.stack = Mock(return_value=Mock())

        # Import after mocking
        from sam3_video.memory_bank import MemoryBank
        self.MemoryBank = MemoryBank

    def test_initialization(self):
        """Test MemoryBank initialization."""
        bank = self.MemoryBank(
            max_recent_frames=8,
            max_prompted_frames=4
        )

        self.assertEqual(bank.max_recent, 8)
        self.assertEqual(bank.max_prompted, 4)
        self.assertEqual(len(bank.recent_frames), 0)
        self.assertEqual(len(bank.prompted_frames), 0)
        self.assertEqual(bank.total_frames_processed, 0)

    def test_add_frame_recent_only(self):
        """Test adding frame to recent queue only."""
        bank = self.MemoryBank()

        # Mock feature array
        features = Mock()
        features.ndim = 3
        features.shape = [256, 64, 64]

        bank.add_frame(
            frame_id=0,
            features=features,
            has_prompt=False
        )

        self.assertEqual(len(bank.recent_frames), 1)
        self.assertEqual(len(bank.prompted_frames), 0)
        self.assertEqual(bank.total_frames_processed, 1)

    def test_add_frame_with_prompt(self):
        """Test adding frame with prompt."""
        bank = self.MemoryBank()

        features = Mock()
        features.ndim = 3
        features.shape = [256, 64, 64]

        bank.add_frame(
            frame_id=0,
            features=features,
            has_prompt=True
        )

        self.assertEqual(len(bank.recent_frames), 1)
        self.assertEqual(len(bank.prompted_frames), 1)

    def test_fifo_behavior(self):
        """Test FIFO queue behavior."""
        bank = self.MemoryBank(max_recent_frames=3)

        features = Mock()
        features.ndim = 3
        features.shape = [256, 64, 64]

        # Add 5 frames (more than max)
        for i in range(5):
            bank.add_frame(frame_id=i, features=features)

        # Should only keep last 3
        self.assertEqual(len(bank.recent_frames), 3)

        # Check frame IDs (should be 2, 3, 4)
        frame_ids = [f['frame_id'] for f in bank.recent_frames]
        self.assertEqual(frame_ids, [2, 3, 4])

    def test_clear(self):
        """Test clearing memory bank."""
        bank = self.MemoryBank()

        features = Mock()
        features.ndim = 3
        features.shape = [256, 64, 64]

        bank.add_frame(frame_id=0, features=features)
        bank.clear()

        self.assertEqual(len(bank.recent_frames), 0)
        self.assertEqual(len(bank.prompted_frames), 0)

    def test_get_memory_stats(self):
        """Test memory statistics."""
        bank = self.MemoryBank(max_recent_frames=10)

        features = Mock()
        features.ndim = 3
        features.shape = [256, 64, 64]

        # Add 5 frames
        for i in range(5):
            bank.add_frame(frame_id=i, features=features)

        stats = bank.get_memory_stats()

        self.assertEqual(stats['recent_frames_count'], 5)
        self.assertEqual(stats['total_frames_processed'], 5)
        self.assertEqual(stats['max_recent'], 10)
        self.assertAlmostEqual(stats['memory_utilization_recent'], 0.5)


class TestMemoryBankConfig(unittest.TestCase):
    """Test MemoryBankConfig presets."""

    def setUp(self):
        """Set up mocks."""
        # Mock all dependencies
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()

        from sam3_video.memory_bank import MemoryBankConfig
        self.MemoryBankConfig = MemoryBankConfig

    def test_default_config(self):
        """Test default configuration."""
        config = self.MemoryBankConfig.default()

        self.assertEqual(config['max_recent_frames'], 8)
        self.assertEqual(config['max_prompted_frames'], 4)

    def test_low_memory_config(self):
        """Test low memory configuration."""
        config = self.MemoryBankConfig.low_memory()

        self.assertEqual(config['max_recent_frames'], 4)
        self.assertEqual(config['max_prompted_frames'], 2)

    def test_high_accuracy_config(self):
        """Test high accuracy configuration."""
        config = self.MemoryBankConfig.high_accuracy()

        self.assertEqual(config['max_recent_frames'], 16)
        self.assertEqual(config['max_prompted_frames'], 8)


class TestModuleStructure(unittest.TestCase):
    """Test module structure and organization."""

    def test_module_exports(self):
        """Test that module exports correct classes."""
        # Mock dependencies
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()

        import sam3_video

        # Check __all__ exports
        self.assertIn('MemoryBank', sam3_video.__all__)
        self.assertIn('Sam3VideoSessionManager', sam3_video.__all__)

    def test_version(self):
        """Test module version."""
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()

        import sam3_video

        self.assertTrue(hasattr(sam3_video, '__version__'))
        self.assertIsInstance(sam3_video.__version__, str)


class TestFileStructure(unittest.TestCase):
    """Test file structure and paths."""

    def test_core_files_exist(self):
        """Test that all core files exist."""
        base_path = Path(__file__).parent.parent

        required_files = [
            'sam3_video/__init__.py',
            'sam3_video/memory_bank.py',
            'sam3_video/session_manager.py',
            'examples/video_tracking_demo.py',
            'examples/webcam_demo.py',
            'examples/simple_example.py',
            'benchmarks/video_benchmark.py',
            'scripts/build_video_engines.sh',
            'onnxexport.py',
            'onnxexport_video.py',
            'requirements.txt',
            'README.md',
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            self.assertTrue(
                full_path.exists(),
                f"Required file not found: {file_path}"
            )

    def test_scripts_executable(self):
        """Test that shell scripts are executable."""
        base_path = Path(__file__).parent.parent
        script_path = base_path / 'scripts' / 'build_video_engines.sh'

        import os
        self.assertTrue(
            os.access(script_path, os.X_OK),
            "build_video_engines.sh is not executable"
        )


class TestImportSafety(unittest.TestCase):
    """Test that imports are safe and won't cause issues."""

    def test_no_circular_imports(self):
        """Test for circular import issues."""
        # Mock all dependencies
        sys.modules['numpy'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['cv2'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['transformers'] = MagicMock()

        # This should not raise
        try:
            import sam3_video
            from sam3_video import MemoryBank
            from sam3_video import Sam3VideoSessionManager
        except ImportError as e:
            if "circular import" in str(e).lower():
                self.fail(f"Circular import detected: {e}")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryBank))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryBankConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestImportSafety))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("SAM3 Video Tracking - Unit Tests")
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

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
