"""Runs evaluation on chess games and loads them to database and TFRecord."""

import concurrent
import contextlib
import io
import os
import sqlite3
from enum import Enum
from logging import CRITICAL, DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

basicConfig(
    filename="eval.log",
    level=DEBUG,
    format="%(asctime)s:%(filename)s:%(funcName)s:[%(levelname)s]: %(message)s",
    force=True,
    filemode="w",
)
logger = getLogger(__name__)

# Disable all logs from chess.engine
getLogger("chess.engine").setLevel(CRITICAL)

# Silence asyncio DEBUG logs
getLogger("asyncio").setLevel(INFO)

# Set memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    logger.info(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)  # noqa: FBT003
        logger.info(f"Memory growth set to True for {device}")
else:
    logger.warning("No GPU found, using CPU")

# Use mixed precision to reduce memory usage.
try:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Using mixed precision policy")
except Exception:  # noqa: BLE001
    logger.warning("Mixed precision not supported or enabled")


class Errors(Enum):
    NONE = 0
    INACCURACY = 1
    MISTAKE = 2
    BLUNDER = 3

    @property
    def threshold(self) -> float:
        thresh = {
            Errors.BLUNDER: 0.3,
            Errors.MISTAKE: 0.2,
            Errors.INACCURACY: 0.1,
        }
        return thresh[self]


class ChessAnalyzer:
    MIN_PLY_COUNT = 5
    RAPID_THRESH = 1499

    def __init__(
        self,
        db_path: Path,
        sf_path: Path = "stockfish",
        depth: int = 18,
        threads: int = 4,
        max_workers: int | None = None,
    ):
        """Initialize the chess analyzer.

        Args:
            db_path (str): Path to the SQLite database
            stockfish_path (str): Path to the Stockfish executable
            depth (int): Analysis depth for Stockfish
            threads (int): Number of threads for Stockfish to use
            max_workers (int): Maximum number of parallel workers
        """
        self.db_path = db_path
        self.stockfish_path = sf_path
        self.depth = depth
        self.engine_threads = threads
        self.max_workers = max_workers or os.cpu_count()

        try:
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.quit()
            logger.debug(f"Stockfish found at {sf_path}")
        except Exception:
            logger.exception("Error initializing Stockfish")
            logger.debug("Please provide a valid path to Stockfish executable")
            raise

        self._init_analysis_tables()

    def _init_analysis_tables(self) -> None:
        """Create the analysis tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Game level evaluations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_game_level (
            game_id INTEGER PRIMARY KEY,
            time_control TEXT,          -- Time control of the game
            estimated_time INTEGER,     -- Estimated game time in seconds
            game_type TEXT,             -- Game type (rapid, blitz, classical, etc.)
            black_blunders INTEGER,     -- Number of blunders made by black
            black_mistakes INTEGER,     -- Number of mistakes made by black
            black_inaccuracies INTEGER, -- Number of inaccuracies made by black
            white_blunders INTEGER,     -- Number of blunders made by white
            white_mistakes INTEGER,     -- Number of mistakes made by white
            white_inaccuracies INTEGER, -- Number of inaccuracies made by white
            analysis_completed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Move level evaluations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_move_level (
            game_id INTEGER,
            halfmove_count INTEGER,
            move TEXT,
            fen TEXT,
            turn INTEGER,               -- 0 for white, 1 for black
            error INTEGER,              -- 0: none, 1: inaccuracy, 2: mistake, 3: blunder
            cp INTEGER,                 -- Centipawn evaluation
            mate INTEGER,               -- Mate in X moves (NULL if no mate)
            time_ratio REAL,            -- Ratio of time spent on this move
            winning_chance REAL,        -- Probability of winning
            drawing_chance REAL,        -- Probability of drawing
            losing_chance REAL,         -- Probability of losing
            is_check INTEGER,           -- 1 if check, 0 otherwise
            is_checkmate INTEGER,       -- 1 if checkmate, 0 otherwise
            PRIMARY KEY (game_id, halfmove_count),
            FOREIGN KEY (game_id) REFERENCES evaluation_game_level(game_id)
        )
        """)

        conn.commit()

    def evaluate_move(self, wdl_before: chess.engine.Wdl, wdl_after: chess.engine.Wdl) -> Errors:
        before_expected_score = wdl_before[0] + (0.5 * wdl_before[1])
        after_expected_score = wdl_after[2] + (0.5 * wdl_after[1])

        score_drop = (before_expected_score - after_expected_score) / 1000

        if score_drop >= Errors.BLUNDER.threshold:
            return Errors.BLUNDER
        if score_drop >= Errors.MISTAKE.threshold:
            return Errors.MISTAKE
        if score_drop >= Errors.INACCURACY.threshold:
            return Errors.INACCURACY

        return Errors.NONE

    def parse_time_control(self, time_control_str: str) -> tuple:
        """Parse the time control string and calculate estimated game time.

        Args:
            time_control_str (str): Time control string (e.g., "180+0", "300+2")

        Returns:
            tuple: (base_time, increment, estimated_time)
        """
        if not time_control_str or time_control_str == "-":
            return (None, None, None)

        try:
            # Handle standard time control format "base+increment"
            if "+" in time_control_str:
                parts = time_control_str.split("+")
                base_time = int(parts[0])
                increment = int(parts[1])

                # Calculate estimated time: base_time + (40 * increment)
                estimated_time = base_time + (40 * increment)

                return (base_time, increment, estimated_time)
            # Handle time formats without increment
            base_time = int(time_control_str)
        except Exception:
            logger.exception(f"Error parsing time control '{time_control_str}'")
            return (None, None, None)
        else:
            return (base_time, 0, base_time)

    def analyze_game(self, game_data: tuple[str]) -> dict[Any]:
        """Analyze a single chess game using Stockfish.

        Args:
            game_data (tuple): Game data from the database

        Returns:
            dict: Analysis results
        """
        game_id = game_data[0]
        pgn_text = game_data[14]
        time_control = game_data[10]
        estimated_time = game_data[16]
        game_type = game_data[17]
        white_elo = game_data[6]
        black_elo = game_data[8]

        if "1/2" in game_data[9]:
            result = 0
        elif game_data[9].startswith("1"):
            result = 1
        elif game_data[9].startswith("0"):
            result = -1

        _, increment, _ = self.parse_time_control(time_control)

        if not pgn_text or pgn_text == "":
            logger.warning("No PGN data")
            return {
                "game_id": game_id,
                "error": "No PGN data",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        try:
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)

            if game is None:
                logger.warning("Failed to parse PGN")
                return {
                    "game_id": game_id,
                    "error": "Failed to parse PGN",
                    "time_control": time_control,
                    "estimated_time": estimated_time,
                    "game_type": game_type,
                }

            if game.end().ply() / 2 < self.MIN_PLY_COUNT:
                logger.warning(f"Game: ({game_id}) too short!")
                return {
                    "game_id": game_id,
                    "error": "Game too short",
                    "time_control": time_control,
                    "estimated_time": estimated_time,
                    "game_type": game_type,
                }
        except Exception:
            logger.exception("Error parsing PGN")
            return {
                "game_id": game_id,
                "error": "Error parsing PGN",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        # Initialize Stockfish
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            engine.configure(
                {
                    "Threads": self.engine_threads,
                    "Skill Level": 20,  # Max strength
                    "UCI_LimitStrength": False,
                    "UCI_ShowWDL": True,
                },
            )
        except Exception:
            logger.exception("Error initializing engine.")
            return {
                "game_id": game_id,
                "error": "Error initializing engine",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        try:
            board = game.board()

            move_features = []

            white_blunders = 0
            white_mistakes = 0
            white_inaccuracies = 0
            black_blunders = 0
            black_mistakes = 0
            black_inaccuracies = 0

            curr_node = game
            prev_node = chess.pgn.Game()

            prev_prev_clock = None

            turn = int(chess.BLACK)

            halfmove_count = -1

            while curr_node:
                error = Errors.NONE
                move = curr_node.move

                if curr_node.parent is not None:
                    board.push(move)

                info = engine.analyse(board, chess.engine.Limit(depth=self.depth), info=chess.engine.Info.SCORE)

                score = info["score"].white()
                cp = score.score(mate_score=1000)
                mate = score.mate()
                time_spent = (
                    0 if (not curr_node.parent) or (not prev_prev_clock) else prev_prev_clock - curr_node.clock()
                )
                time_ratio = (
                    0 if (not prev_node.parent) or (not prev_prev_clock) else time_spent / (prev_prev_clock + increment)
                )
                halfmove_count += 1
                turn = int(not turn)
                is_check = board.is_check()
                is_checkmate = board.is_checkmate()

                try:
                    curr_node.wdl = info["wdl"]
                except KeyError:
                    wdl = chess.engine.Wdl(1000, 0, 0)
                    curr_node.wdl = chess.engine.PovWdl(wdl, not turn)

                if curr_node.parent:
                    error = self.evaluate_move(prev_node.wdl, curr_node.wdl)
                    if error is Errors.BLUNDER:
                        if turn is int(chess.BLACK):
                            white_blunders += 1
                        else:
                            black_blunders += 1
                    if error is Errors.MISTAKE:
                        if turn is int(chess.BLACK):
                            white_mistakes += 1
                        else:
                            black_mistakes += 1
                    if error is Errors.INACCURACY:
                        if turn is int(chess.BLACK):
                            white_inaccuracies += 1
                        else:
                            black_inaccuracies += 1

                features = {
                    "halfmove_count": halfmove_count,
                    "move": move,
                    "fen": board.fen(),
                    "turn": turn,
                    "error": error.value,
                    "cp": cp,
                    "mate": mate,
                    "time_ratio": time_ratio,
                    "winning_chance": curr_node.wdl.white().winning_chance(),
                    "drawing_chance": curr_node.wdl.white().drawing_chance(),
                    "losing_chance": curr_node.wdl.white().losing_chance(),
                    "is_check": int(is_check),
                    "is_checkmate": int(is_checkmate),
                }

                move_features.append(features)

                prev_prev_clock, prev_node, curr_node = prev_node.clock(), curr_node, curr_node.next()

            # Close the engine
            engine.quit()

        except Exception:
            # Make sure to quit the engine if an error occurs
            with contextlib.suppress(Exception):
                engine.quit()

            logger.exception("Error analyzing game.")

            return {
                "game_id": game_id,
                "error": "Error analyzing game",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }
        else:
            return {
                "game_id": game_id,
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
                "black_errors": (black_blunders, black_mistakes, black_inaccuracies),
                "white_errors": (white_blunders, white_mistakes, white_inaccuracies),
                "outcome": result,
                "moves": move_features,
                "target": (white_elo, black_elo),
                "total_move_count": halfmove_count,
            }

    def get_games_to_analyze(
        self,
        connection: sqlite3.Connection,
        limit: Optional[int] = None,
        game_type_filter: str = "rapid",
        last_game_id: int = 0,
    ) -> sqlite3.Cursor:
        """Get games that haven't been analyzed yet, filtered by game type.

        Args:
            connection (sqlite3.Connection): Database connection
            limit (int, optional): Maximum number of games to retrieve
            game_type_filter (str): Type of games to filter for (rapid, blitz, classical, etc.)
            last_game_id (int): ID of the last game processed for keyset pagination

        Returns:
            cursor: Cursor for the executed query.
        """
        cursor = connection.cursor()

        query = """
        SELECT g.* FROM game_with_type g
        LEFT JOIN evaluation_game_level a ON g.ID = a.game_id
        WHERE a.game_id IS NULL AND g.result IS NOT NULL
        """

        # Apply game type filter
        if game_type_filter:
            query += f" AND g.game_type = '{game_type_filter}' AND (g.MOVES != '1-0\n' AND g.MOVES != '0-1\n')"

        # Apply offset for resuming analysis
        if last_game_id > 0:
            query += f" AND g.ID > {last_game_id}"

        # Add ORDER BY to ensure consistent results when using offset
        query += " ORDER BY g.ID"

        # Apply limit if specified
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return cursor

    def save_to_database(self, analysis_results: list[dict[str, Any]]) -> None:
        """Save analysis results to the database.

        Args:
            analysis_results (List[Dict]): List of analysis results from analyzed games
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            conn.execute("BEGIN TRANSACTION")

            for result in analysis_results:
                if "error" in result:
                    logger.warning(f"Skipping game {result['game_id']} due to error: {result['error']}")
                    continue

                # Insert game level data
                game_id = result["game_id"]
                black_blunders, black_mistakes, black_inaccuracies = result["black_errors"]
                white_blunders, white_mistakes, white_inaccuracies = result["white_errors"]

                cursor.execute(
                    """
                INSERT INTO evaluation_game_level
                (game_id, time_control, estimated_time, game_type,
                black_blunders, black_mistakes, black_inaccuracies,
                white_blunders, white_mistakes, white_inaccuracies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        game_id,
                        result["time_control"],
                        result["estimated_time"],
                        result["game_type"],
                        black_blunders,
                        black_mistakes,
                        black_inaccuracies,
                        white_blunders,
                        white_mistakes,
                        white_inaccuracies,
                    ),
                )

                # Insert move level data
                move_data = []
                for move in result["moves"]:
                    move_data.append(
                        (
                            game_id,
                            move["halfmove_count"],
                            str(move["move"]),
                            move["fen"],
                            move["turn"],
                            move["error"],
                            move["cp"],
                            move["mate"],
                            move["time_ratio"],
                            move["winning_chance"],
                            move["drawing_chance"],
                            move["losing_chance"],
                            move["is_check"],
                            move["is_checkmate"],
                        ),
                    )

                cursor.executemany(
                    """
                INSERT INTO evaluation_move_level
                (game_id, halfmove_count, move, fen, turn, error, cp, mate,
                time_ratio, winning_chance, drawing_chance, losing_chance,
                is_check, is_checkmate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    move_data,
                )

            conn.commit()
            logger.info(f"Successfully saved {len(analysis_results)} games to database")

        except Exception:
            conn.rollback()
            logger.exception("Error saving to database.")
        finally:
            conn.close()

    def _create_tf_feature(self, value: Any) -> Any:
        """Create appropriate TensorFlow feature from a value."""
        if isinstance(value, int):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        if isinstance(value, float):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        if isinstance(value, str):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))
        if isinstance(value, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        raise ValueError(f"Unsupported type: {type(value)}")

    def _create_int64_list_feature(self, values):
        """Helper method for creating int64 list features."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _create_float_list_feature(self, values):
        """Helper method for creating float list features."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def normalize_estimated_time(self, est_t: int) -> float:
        return est_t / self.RAPID_THRESH

    def normalize_by_max_move(self, count: int, total_move_count: int) -> float:
        """Normalize scalar features."""
        # scalar_scaler = MinMaxScaler(feature_range=(0, 1))
        return count / total_move_count

    def encode_move(self, move_obj: chess.Move) -> np.array:
        """Encode a python-chess move to a normalized 5D vector.

        [from_col, from_row, to_col, to_row, promotion]
        All values are scaled to [0, 1] for Transformer compatibility.
        """
        # move_obj = chess.Move.from_uci(move_obj) # uncomment if want to use string
        if move_obj is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        from_sq = move_obj.from_square
        to_sq = move_obj.to_square

        from_col = chess.square_file(from_sq) / 7.0
        from_row = chess.square_rank(from_sq) / 7.0
        to_col = chess.square_file(to_sq) / 7.0
        to_row = chess.square_rank(to_sq) / 7.0

        # Normalize promotion: 0 if not a promotion, else map [1, 2, 3, 4, 5] → [0.2, ..., 1.0]
        # Promotion types: None (0), Knight (2), Bishop (3), Rook (4), Queen (5)
        promotion_raw = move_obj.promotion if move_obj.promotion is not None else 0
        promotion_normalized = promotion_raw / 5.0  # Max promotion code is 5 (Queen)
        return np.array([from_col, from_row, to_col, to_row, promotion_normalized], dtype=np.float32)

    def encode_fen(self, fen: str) -> np.array:
        """Encode FEN string to include.

        - Signed material values: white +ve, black -ve (normalized to [-1, 1])
        - Castling rights (4 binary flags)
        - En passant square (2 normalized floats)

        Output shape: (64 + 4 + 2) = (70,)
        """
        casteling_index = 2
        en_passent_index = 3

        parts = fen.split(" ")
        board_fen = parts[0]
        castling = parts[2] if len(parts) > casteling_index else "-"
        en_passant = parts[3] if len(parts) > en_passent_index else "-"

        # --- 1. Encode signed board pieces ---
        base_map = {
            "p": -1,
            "n": -2,
            "b": -3,
            "r": -4,
            "q": -5,
            "k": -6,
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
        }
        expanded = ""
        for ch in board_fen:
            if ch.isdigit():
                expanded += " " * int(ch)
            elif ch == "/":
                continue
            else:
                expanded += ch

        # Map to values in [-1, 1] (divide by max absolute value 6)
        board_encoded = [base_map.get(ch, 0) / 6.0 if ch != " " else 0.0 for ch in expanded]

        # --- 2. Castling rights: [K, Q, k, q]
        castling_flags = [
            1.0 if "K" in castling else 0.0,
            1.0 if "Q" in castling else 0.0,
            1.0 if "k" in castling else 0.0,
            1.0 if "q" in castling else 0.0,
        ]

        # --- 3. En passant square: normalized col and row
        if en_passant != "-" and len(en_passant) == 2:
            col = ord(en_passant[0]) - ord("a")  # 0-7
            row = int(en_passant[1]) - 1  # 0-7
            en_passant_encoded = [col / 7.0, row / 7.0]
        else:
            en_passant_encoded = [0.0, 0.0]

        return np.array(board_encoded + castling_flags + en_passant_encoded, dtype=np.float32)

    def create_mate_scaler(self, min_n: int = 0, max_n: int = 20) -> MinMaxScaler:
        """Prepares a MinMaxScaler using 1 / exp(0.1 * mate_in_n) for mate_in_n in range(min_n, max_n)."""
        mate_values = np.array([1 / np.exp(0.1 * n) for n in range(min_n, max_n + 1)]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaler.fit(mate_values)
        return scaler

    def normalize_eval_and_mate(self, eval_score: chess.engine.Score, mate_in_n: Optional[int] = None):
        """Normalize chess eval and mate-in-N.

        Returns:
            normalized_eval: [-0.99, 0.99], or ±1.0 if mate detected.
            normalized_mate: [0.0, 1.0], 0.0 if mate_in_n > 20.
        """
        eval_scale = 1500.0
        mate_pad_thresh = 20

        mate_scaler = self.create_mate_scaler()

        if mate_in_n is not None:
            sign = np.sign(eval_score)
            normalized_eval = sign * 1.0

            if mate_in_n > mate_pad_thresh:
                normalized_mate = 0.0
            else:
                raw_value = np.array([[1 / np.exp(0.1 * mate_in_n)]])
                normalized_mate = mate_scaler.transform(raw_value)[0][0]
        else:
            normalized_eval = np.tanh(eval_score / eval_scale) * 0.99
            normalized_mate = 0.0

        return normalized_eval, normalized_mate

    def export_to_tfrecord(self, analysis_results: list[dict[str, Any]], output_path: str) -> None:
        """Export analysis results to TFRecord format.

        Args:
            analysis_results (List[Dict]): List of analysis results from analyzed games
            output_path (str): Path to save the TFRecord file
        """
        logger.info(f"Exporting {len(analysis_results)} games to TFRecord at {output_path}")

        try:
            with tf.io.TFRecordWriter(output_path) as writer:
                for result in analysis_results:
                    if "error" in result:
                        continue

                    total_move_count = result["total_move_count"]
                    white_elo, black_elo = result["target"]
                    normalized_est = self.normalize_estimated_time(result["estimated_time"])
                    black_blunders, black_mistakes, black_inaccuracies = result["black_errors"]
                    white_blunders, white_mistakes, white_inaccuracies = result["white_errors"]

                    # Game level features
                    game_features = {
                        "estimated_time": self._create_tf_feature(normalized_est),
                        "white_elo": self._create_tf_feature(white_elo),
                        "black_elo": self._create_tf_feature(black_elo),
                        "outcome": self._create_tf_feature(result["outcome"]),
                        "total_move_count": self._create_tf_feature(total_move_count),
                        # Error counts
                        "black_blunders": self._create_tf_feature(
                            self.normalize_by_max_move(black_blunders, total_move_count),
                        ),
                        "black_mistakes": self._create_tf_feature(
                            self.normalize_by_max_move(black_mistakes, total_move_count),
                        ),
                        "black_inaccuracies": self._create_tf_feature(
                            self.normalize_by_max_move(black_inaccuracies, total_move_count),
                        ),
                        "white_blunders": self._create_tf_feature(
                            self.normalize_by_max_move(white_blunders, total_move_count),
                        ),
                        "white_mistakes": self._create_tf_feature(
                            self.normalize_by_max_move(white_mistakes, total_move_count),
                        ),
                        "white_inaccuracies": self._create_tf_feature(
                            self.normalize_by_max_move(white_inaccuracies, total_move_count),
                        ),
                    }

                    # Move level features
                    if "moves" in result:
                        move_features = result["moves"]

                        halfmove_count = []
                        moves = []
                        fens = []
                        is_errors = []
                        cp = []
                        mate_in_n = []
                        time_ratios = []
                        winning_chance = []
                        drawing_chance = []
                        losing_chance = []
                        check_flag = []
                        mate_flag = []
                        turn = []

                        for move_data in move_features:
                            halfmove_count.append(
                                self.normalize_by_max_move(move_data["halfmove_count"], total_move_count),
                            )
                            moves.extend(self.encode_move(move_data["move"]))
                            fens.extend(self.encode_fen(move_data["fen"]))
                            is_errors.append(move_data["error"] / 3)

                            normalized_score, normalized_mate = self.normalize_eval_and_mate(
                                move_data["cp"],
                                move_data["mate"],
                            )
                            cp.append(normalized_score)
                            mate_in_n.append(normalized_mate)

                            time_ratios.append(move_data["time_ratio"])
                            winning_chance.append(move_data["winning_chance"])
                            drawing_chance.append(move_data["drawing_chance"])
                            losing_chance.append(move_data["losing_chance"])

                            check_flag.append(move_data["is_check"])
                            mate_flag.append(move_data["is_checkmate"])

                            turn.append(move_data["turn"])

                        # Add all move-level features to game_features
                        game_features.update(
                            {
                                "halfmove_counts": self._create_float_list_feature(halfmove_count),
                                "encoded_moves": self._create_float_list_feature(moves),
                                "encoded_fens": self._create_float_list_feature(fens),
                                "error_codes": self._create_float_list_feature(is_errors),
                                "cp_scores": self._create_float_list_feature(cp),
                                "time_ratios": self._create_float_list_feature(time_ratios),
                                "mate_in_n": self._create_float_list_feature(mate_in_n),
                                "winning_chances": self._create_float_list_feature(winning_chance),
                                "drawing_chances": self._create_float_list_feature(drawing_chance),
                                "losing_chances": self._create_float_list_feature(losing_chance),
                                "check_flags": self._create_int64_list_feature(check_flag),
                                "mate_flags": self._create_int64_list_feature(mate_flag),
                                "turn": self._create_int64_list_feature(turn),
                            },
                        )

                    with open("test.txt", "a") as file:
                        file.writelines(str(game_features))

                    example = tf.train.Example(features=tf.train.Features(feature=game_features))
                    writer.write(example.SerializeToString())

            logger.info(f"Successfully exported to {output_path}")

        except Exception:
            logger.exception("Error exporting to TFRecord.")

    def run_analysis(
        self,
        batch_size: int = 100,
        total_games: Optional[int] = None,
        game_type_filter: str = "rapid",
        resume_id: int = 0,
        tfrecord_dir: str = "tfrecords",
    ) -> None:
        """Run analysis on unanalyzed games in parallel.

        Args:
            batch_size (int): Number of games to process in each batch
            total_games (int, optional): Total number of games to process
            game_type_filter (str): Type of games to analyze (rapid, blitz, classical, etc.)
            start_offset (int, optional): Initial offset for resuming a previous analysis run
            tfrecord_dir (str): Directory to save TFRecord files
        """
        games_processed = 0
        last_game_id = resume_id
        batch_counter = 1

        # Create TFRecord directory if it doesn't exist
        os.makedirs(tfrecord_dir, exist_ok=True)

        logger.info(f"Starting analysis of {game_type_filter} games from ID: {last_game_id}...")

        while True:
            # Use a separate connection for fetching games since we'll be passing them to processes
            conn = sqlite3.connect(self.db_path)
            cursor = self.get_games_to_analyze(
                conn,
                limit=batch_size,
                game_type_filter=game_type_filter,
                last_game_id=last_game_id,
            )

            games = cursor.fetchall()
            conn.close()

            if not games:
                logger.warning(f"No more {game_type_filter} games to analyze")
                break

            if total_games and games_processed >= total_games:
                logger.info(f"Reached target of {total_games} games")
                break

            games_count = len(games)
            logger.debug(
                f"Processing batch of {games_count} {game_type_filter} games (last game ID: {last_game_id})...",
            )

            # Process games in parallel using ProcessPoolExecutor for CPU-bound tasks
            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # We need to provide all necessary instance variables to analyze_game
                # since it will be executed in a separate process
                future_to_game = {
                    executor.submit(
                        self.analyze_game,
                        game,
                    ): game
                    for game in games
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_game), total=len(games)):
                    try:
                        game_result = future.result()
                        results.append(game_result)

                    except Exception:
                        game_id = future_to_game[future][0]  # First element is typically game_id
                        logger.exception(f"Game {game_id} generated an exception.")

            clean_results = []
            error_games = []
            for game in results:
                if "error" not in game:
                    clean_results.append(game)
                else:
                    error_games.append(game)

            logger.debug(f"Clean Games: {[game['game_id'] for game in clean_results]}")
            logger.warning(f"Error Games: {[game['game_id'] for game in error_games]}")

            # Save results to database
            logger.info(f"Saving batch {batch_counter} results to database...")
            self.save_to_database(clean_results)

            # Export to TFRecord
            tfrecord_path = os.path.join(tfrecord_dir, f"{game_type_filter}_batch_{last_game_id}.tfrecord")
            logger.info(f"Exporting batch {batch_counter} to TFRecord...")
            self.export_to_tfrecord(clean_results, tfrecord_path)

            if games:
                last_game_id = games[-1][0]
                logger.debug(f"Last Game ID: {last_game_id}")

            # Update counters
            games_processed += len(clean_results)
            batch_counter += 1

            logger.info(f"Completed batch {batch_counter - 1}. Total games processed: {games_processed}")

        logger.info(f"Analysis complete. Processed {games_processed} games.")


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    # analyzer = ChessAnalyzer(
    #     db_path="/home/vandy/work/chess/data/db.ocgdb.db3",
    #     stockfish_path="/home/vandy/.local/bin/stockfish",  # Path to Stockfish executable
    #     depth=14,  # Analysis depth
    #     threads=2,  # Threads per engine
    #     max_workers=8,  # Number of parallel processes
    # )

    analyzer = ChessAnalyzer(
        Path("/home/vandy/work/chess/data/db.ocgdb.db3"),
        Path("/home/vandy/.local/bin/stockfish"),
        depth=14,
        threads=2,
        max_workers=8,
    )

    # Run analysis
    # NOTE: 150554 -> 184254 -> 217773 -> 250837 already processed.
    analyzer.run_analysis(5000, total_games=20000, tfrecord_dir="data/tfrecords", resume_id=140673)
    # analyzer.run_analysis(batch_size=50, total_games=100000)

    # Generate visualizations
    # analyzer.visualize_results(limit=100)
