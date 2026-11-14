"""20-plane encoder + 13→20 warm-start padding."""

import unittest

import chess
import torch

from tabula.checkpoint import adapt_model_state_dict, load_model_weights
from tabula.game.encoder import INPUT_PLANES, encode_board
from tabula.model.model import TabulaNet


class EncoderPlaneTests(unittest.TestCase):
    def test_encoder_shape_is_20_planes(self) -> None:
        board = chess.Board()
        tensor = encode_board(board)
        self.assertEqual(tensor.shape, (INPUT_PLANES, 8, 8))
        self.assertEqual(str(tensor.dtype), "float32")
        self.assertEqual(float(tensor[12].mean()), 1.0)  # White to move
        self.assertEqual(float(tensor[19].mean()), 0.0)  # halfmove 0
        # Castling rights all available for White/Black at start.
        self.assertEqual(float(tensor[15].mean()), 1.0)
        self.assertEqual(float(tensor[16].mean()), 1.0)
        self.assertEqual(float(tensor[17].mean()), 1.0)
        self.assertEqual(float(tensor[18].mean()), 1.0)

    def test_repetition_and_halfmove_planes(self) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            board.push_uci(uci)
        # Position after 4 plies is the start again with White to move — seen twice?
        # Start occurred at ply 0 and again now → is_repetition(2) True.
        self.assertTrue(board.is_repetition(2))
        tensor = encode_board(board)
        self.assertEqual(float(tensor[13].mean()), 1.0)
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8")
        self.assertTrue(board.is_repetition(3))
        tensor3 = encode_board(board)
        self.assertEqual(float(tensor3[14].mean()), 1.0)

    def test_halfmove_plane_normalizes(self) -> None:
        board = chess.Board()
        board.halfmove_clock = 50
        tensor = encode_board(board)
        self.assertAlmostEqual(float(tensor[19].mean()), 0.5, places=5)
        board.halfmove_clock = 150
        tensor = encode_board(board)
        self.assertAlmostEqual(float(tensor[19].mean()), 1.0, places=5)


class WarmStartTests(unittest.TestCase):
    def test_13_to_20_channel_padding(self) -> None:
        old = TabulaNet(num_res_blocks=2, input_planes=13)
        new = TabulaNet(num_res_blocks=2, input_planes=20)
        old_state = old.state_dict()
        adapted, padded = adapt_model_state_dict(new, old_state)
        self.assertTrue(padded)
        self.assertEqual(adapted["start_block.0.weight"].shape[1], 20)
        # First 13 channels preserved.
        torch.testing.assert_close(
            adapted["start_block.0.weight"][:, :13],
            old_state["start_block.0.weight"],
        )
        # New channels zero.
        self.assertTrue(torch.all(adapted["start_block.0.weight"][:, 13:] == 0))

        loaded = TabulaNet(num_res_blocks=2, input_planes=20)
        self.assertTrue(load_model_weights(loaded, old_state))
        out = loaded(torch.zeros(1, 20, 8, 8))
        self.assertEqual(out[0].shape[-1], 4672)


if __name__ == "__main__":
    unittest.main()
