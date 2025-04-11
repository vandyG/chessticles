import concurrent.futures
import io
import json
import os
import re
import sqlite3
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class ChessAnalyzer:
    def __init__(self, db_path, stockfish_path="stockfish", depth=18, threads=4, max_workers=None):
        """Initialize the chess analyzer.

        Args:
            db_path (str): Path to the SQLite database
            stockfish_path (str): Path to the Stockfish executable
            depth (int): Analysis depth for Stockfish
            threads (int): Number of threads for Stockfish to use
            max_workers (int): Maximum number of parallel workers
        """
        self.db_path = db_path
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.engine_threads = threads
        self.max_workers = max_workers or os.cpu_count()

        # Validate that Stockfish is available
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            engine.quit()
            print(f"Stockfish found at {stockfish_path}")
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            print("Please provide a valid path to Stockfish executable")
            raise

        # Create a table for analysis results if it doesn't exist
        self._init_analysis_table()

    def _init_analysis_table(self):
        """Create the analysis table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_analysis (
            game_id INTEGER PRIMARY KEY,
            white_acl REAL,             -- Average centipawn loss for white
            black_acl REAL,             -- Average centipawn loss for black
            white_blunders INTEGER,     -- Number of blunders by white
            white_mistakes INTEGER,     -- Number of mistakes by white
            white_inaccuracies INTEGER, -- Number of inaccuracies by white
            black_blunders INTEGER,     -- Number of blunders by black
            black_mistakes INTEGER,     -- Number of mistakes by black
            black_inaccuracies INTEGER, -- Number of inaccuracies by black
            time_eval_data BLOB,        -- JSON data for time vs eval correlation
            time_control TEXT,          -- Time control of the game
            estimated_time INTEGER,     -- Estimated game time in seconds
            game_type TEXT,             -- Game type (rapid, blitz, classical, etc.)
            analysis_completed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()

    def parse_time_control(self, time_control_str):
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
            return (base_time, 0, base_time)
        except Exception as e:
            print(f"Error parsing time control '{time_control_str}': {e}")
            return (None, None, None)

    def get_game_type(self, estimated_time):
        """Determine game type based on estimated time.

        Args:
            estimated_time (int): Estimated game time in seconds

        Returns:
            str: Game type (bullet, blitz, rapid, classical)
        """
        if estimated_time is None:
            return "unknown"

        # Game type classification based on estimated time
        if estimated_time <= 179:
            return "bullet"
        if estimated_time <= 479:
            return "blitz"
        if estimated_time <= 1499:
            return "rapid"
        return "classical"

    def get_games_to_analyze(self, limit=None, game_type_filter="rapid"):
        """Get games that haven't been analyzed yet, filtered by game type.

        Args:
            limit (int, optional): Maximum number of games to retrieve
            game_type_filter (str): Type of games to filter for (rapid, blitz, classical, etc.)

        Returns:
            list: List of game tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create a temporary view with game type calculation
        cursor.execute("""
        CREATE TEMPORARY VIEW IF NOT EXISTS game_with_type AS
        SELECT 
            g.*,
            CASE 
                WHEN g.time_control LIKE '%+%' THEN 
                    CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                    (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                    CAST(g.time_control AS INTEGER)
                ELSE NULL
            END AS estimated_time,
            CASE 
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 179 THEN 'bullet'
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 479 THEN 'blitz'
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 1499 THEN 'rapid'
                ELSE 'classical'
            END AS game_type
        FROM games g
        """)

        # Query for games that don't have analysis, filtered by game type
        query = """
        SELECT g.* FROM game_with_type g
        LEFT JOIN game_analysis a ON g.rowid = a.game_id
        WHERE a.game_id IS NULL
        """

        # Apply game type filter
        if game_type_filter:
            query += f" AND g.game_type = '{game_type_filter}'"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        games = cursor.fetchall()

        conn.close()
        return games

    def parse_pgn(self, pgn_text):
        """Parse a PGN string into a chess.pgn.Game object.

        Args:
            pgn_text (str): PGN text of the game

        Returns:
            tuple: (chess.pgn.Game, dict of clock times per move)
        """
        # Handle clock annotations
        clock_times = {}

        # Extract clock times using regex
        clock_pattern = r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]"
        move_num_pattern = r"(\d+)\.\.?\.\s+[^\s]+\s+\{\s*\[%clk"

        move_clocks = []
        for match in re.finditer(clock_pattern, pgn_text):
            hours, minutes, seconds = match.groups()
            time_in_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            move_clocks.append(time_in_seconds)

        # Clean PGN for parsing
        clean_pgn = re.sub(r"\{\s*\[%clk[^}]*\}\s*", "", pgn_text)

        pgn_io = io.StringIO(clean_pgn)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            return None, {}

        # Assign clock times to moves
        if move_clocks:
            move_num = 0
            clock_times = {}

            # Walk through the game to assign clock times
            node = game
            while node.variations:
                node = node.variations[0]
                if move_num < len(move_clocks):
                    clock_times[move_num] = move_clocks[move_num]
                move_num += 1

        return game, clock_times

    def analyze_game(self, game_data):
        """Analyze a single chess game using Stockfish.

        Args:
            game_data (tuple): Game data from the database

        Returns:
            dict: Analysis results
        """
        game_id = game_data[0]
        pgn_text = game_data[14]
        time_control = game_data[10]

        # Parse time control and determine game type
        base_time, increment, estimated_time = self.parse_time_control(time_control)
        game_type = self.get_game_type(estimated_time)

        if not pgn_text or pgn_text == "":
            return {
                "game_id": game_id,
                "error": "No PGN data",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        # Check if PGN already contains analysis
        has_analysis = "eval" in pgn_text.lower() or "evaluation" in pgn_text.lower()

        # Parse the PGN
        try:
            game, clock_times = self.parse_pgn(pgn_text)

            if game is None:
                return {
                    "game_id": game_id,
                    "error": "Failed to parse PGN",
                    "time_control": time_control,
                    "estimated_time": estimated_time,
                    "game_type": game_type,
                }
        except Exception as e:
            return {
                "game_id": game_id,
                "error": f"Error parsing PGN: {e!s}",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        # Initialize Stockfish
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            engine.configure({"Threads": self.engine_threads})
        except Exception as e:
            return {
                "game_id": game_id,
                "error": f"Error initializing engine: {e!s}",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        try:
            # Initialize variables to track analysis
            white_errors = {"blunders": 0, "mistakes": 0, "inaccuracies": 0}
            black_errors = {"blunders": 0, "mistakes": 0, "inaccuracies": 0}
            white_cp_loss = []
            black_cp_loss = []

            # Prepare time-eval correlation data
            time_eval_data = {
                "white": {"times": [], "evals": [], "moves": []},
                "black": {"times": [], "evals": [], "moves": []},
            }

            # Analyze the game
            board = game.board()
            prev_eval = None
            move_num = 0

            # Walk through the game
            node = game
            while node.variations:
                next_node = node.variations[0]
                move = next_node.move

                # Current player (True for white, False for black)
                is_white = board.turn == chess.WHITE
                player_key = "white" if is_white else "black"

                # Get evaluation before the move
                if prev_eval is None:
                    info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                    eval_before = self._get_eval_score(info, board.turn)
                else:
                    eval_before = prev_eval

                # Make the move and get evaluation after
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                eval_after = self._get_eval_score(info, not board.turn)

                # Calculate centipawn loss
                cp_loss = self._calculate_cp_loss(eval_before, eval_after, is_white)

                # Store the loss
                if is_white:
                    white_cp_loss.append(cp_loss)
                    # Classify the move
                    if cp_loss >= 300:  # Blunder
                        white_errors["blunders"] += 1
                    elif cp_loss >= 100:  # Mistake
                        white_errors["mistakes"] += 1
                    elif cp_loss >= 50:  # Inaccuracy
                        white_errors["inaccuracies"] += 1
                else:
                    black_cp_loss.append(cp_loss)
                    # Classify the move
                    if cp_loss >= 300:  # Blunder
                        black_errors["blunders"] += 1
                    elif cp_loss >= 100:  # Mistake
                        black_errors["mistakes"] += 1
                    elif cp_loss >= 50:  # Inaccuracy
                        black_errors["inaccuracies"] += 1

                # Record time and evaluation data if clock info is available
                if move_num in clock_times:
                    time_left = clock_times[move_num]
                    time_eval_data[player_key]["times"].append(time_left)
                    time_eval_data[player_key]["evals"].append(eval_after)
                    time_eval_data[player_key]["moves"].append(move_num)

                # Store the current evaluation for next iteration
                prev_eval = eval_after
                move_num += 1
                node = next_node

            # Calculate average centipawn loss
            white_acl = sum(white_cp_loss) / len(white_cp_loss) if white_cp_loss else 0
            black_acl = sum(black_cp_loss) / len(black_cp_loss) if black_cp_loss else 0

            # Close the engine
            engine.quit()

            # Convert time-eval data to JSON-serializable format
            time_eval_json = json.dumps(time_eval_data)

            # Return the analysis results
            return {
                "game_id": game_id,
                "white_acl": white_acl,
                "black_acl": black_acl,
                "white_blunders": white_errors["blunders"],
                "white_mistakes": white_errors["mistakes"],
                "white_inaccuracies": white_errors["inaccuracies"],
                "black_blunders": black_errors["blunders"],
                "black_mistakes": black_errors["mistakes"],
                "black_inaccuracies": black_errors["inaccuracies"],
                "time_eval_data": time_eval_json,
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

        except Exception as e:
            # Make sure to quit the engine if an error occurs
            try:
                engine.quit()
            except:
                pass

            return {
                "game_id": game_id,
                "error": f"Error analyzing game: {e!s}",
                "time_control": time_control,
                "estimated_time": estimated_time,
                "game_type": game_type,
            }

    def _get_eval_score(self, info, turn):
        """Convert Stockfish evaluation to a numerical score.

        Args:
            info (dict): Stockfish analysis info
            turn (bool): Current player's turn (True for white, False for black)

        Returns:
            float: Evaluation in centipawns from white's perspective
        """
        if "score" not in info:
            return 0

        score = info["score"].relative

        # Handle mate scores
        if score.mate() is not None:
            if score.mate() > 0:
                return 10000 - score.mate() * 10  # Winning
            return -10000 - score.mate() * 10  # Losing

        # Regular centipawn score
        return score.score()

    def _calculate_cp_loss(self, eval_before, eval_after, is_white):
        """Calculate centipawn loss for a move.

        Args:
            eval_before (float): Evaluation before the move
            eval_after (float): Evaluation after the move
            is_white (bool): Whether the player is white

        Returns:
            float: Centipawn loss
        """
        # For white, a decreasing eval is bad
        if is_white:
            cp_loss = max(0, eval_before - eval_after)
        # For black, an increasing eval is bad
        else:
            cp_loss = max(0, eval_after - eval_before)

        return cp_loss

    def save_analysis(self, analysis_results):
        """Save analysis results to the database.

        Args:
            analysis_results (dict): Analysis results
        """
        if "error" in analysis_results:
            print(f"Error for game {analysis_results['game_id']}: {analysis_results['error']}")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
            INSERT INTO game_analysis
            (game_id, white_acl, black_acl, 
             white_blunders, white_mistakes, white_inaccuracies,
             black_blunders, black_mistakes, black_inaccuracies,
             time_eval_data, time_control, estimated_time, game_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    analysis_results["game_id"],
                    analysis_results["white_acl"],
                    analysis_results["black_acl"],
                    analysis_results["white_blunders"],
                    analysis_results["white_mistakes"],
                    analysis_results["white_inaccuracies"],
                    analysis_results["black_blunders"],
                    analysis_results["black_mistakes"],
                    analysis_results["black_inaccuracies"],
                    analysis_results["time_eval_data"],
                    analysis_results["time_control"],
                    analysis_results["estimated_time"],
                    analysis_results["game_type"],
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            conn.close()
            print(f"Error saving analysis for game {analysis_results['game_id']}: {e}")
            return False

    def run_analysis(self, batch_size=100, total_games=None, game_type_filter="rapid"):
        """Run analysis on unanalyzed games in parallel.

        Args:
            batch_size (int): Number of games to process in each batch
            total_games (int, optional): Total number of games to process
            game_type_filter (str): Type of games to analyze (rapid, blitz, classical, etc.)
        """
        games_processed = 0

        print(f"Starting analysis of {game_type_filter} games...")

        # Process games in batches
        while True:
            # Get a batch of games to analyze
            games = self.get_games_to_analyze(limit=batch_size, game_type_filter=game_type_filter)

            if not games:
                print(f"No more {game_type_filter} games to analyze")
                break

            if total_games and games_processed >= total_games:
                print(f"Reached target of {total_games} games")
                break

            print(f"Processing batch of {len(games)} {game_type_filter} games...")

            # Process games in parallel
            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit jobs
                future_to_game = {executor.submit(self.analyze_game, game): game for game in games}

                # Process as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_game), total=len(games)):
                    game = future_to_game[future]
                    try:
                        result = future.result()
                        if "error" not in result:
                            self.save_analysis(result)
                            games_processed += 1
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing game {game[0]}: {e}")

            # Print summary for this batch
            success = sum(1 for r in results if "error" not in r)
            error = sum(1 for r in results if "error" in r)
            print(f"Batch completed: {success} successful, {error} errors")
            print(f"Total {game_type_filter} games processed so far: {games_processed}")

            if total_games and games_processed >= total_games:
                print(f"Reached target of {total_games} {game_type_filter} games")
                break

    def analyze_game_time_distribution(self):
        """Analyze the distribution of game types in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create a temporary view with game type calculation
        cursor.execute("""
        CREATE TEMPORARY VIEW IF NOT EXISTS game_time_analysis AS
        SELECT 
            CASE 
                WHEN g.time_control LIKE '%+%' THEN 
                    CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                    (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                    CAST(g.time_control AS INTEGER)
                ELSE NULL
            END AS estimated_time,
            CASE 
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 179 THEN 'bullet'
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 479 THEN 'blitz'
                WHEN (CASE 
                        WHEN g.time_control LIKE '%+%' THEN 
                            CAST(SUBSTR(g.time_control, 1, INSTR(g.time_control, '+')-1) AS INTEGER) + 
                            (40 * CAST(SUBSTR(g.time_control, INSTR(g.time_control, '+')+1) AS INTEGER))
                        WHEN g.time_control IS NOT NULL AND g.time_control != '-' THEN
                            CAST(g.time_control AS INTEGER)
                        ELSE NULL
                      END) <= 1499 THEN 'rapid'
                ELSE 'classical'
            END AS game_type,
            COUNT(*) as count
        FROM games g
        GROUP BY game_type
        """)

        cursor.execute("SELECT game_type, count FROM game_time_analysis")
        results = cursor.fetchall()

        conn.close()

        # Print and visualize the results
        print("\nGame Type Distribution:")
        total = sum(row[1] for row in results)

        for game_type, count in results:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{game_type}: {count} games ({percentage:.2f}%)")

        # Visualize
        if results:
            plt.figure(figsize=(10, 6))
            game_types = [row[0] or "unknown" for row in results]
            counts = [row[1] for row in results]

            plt.bar(game_types, counts)
            plt.xlabel("Game Type")
            plt.ylabel("Number of Games")
            plt.title("Distribution of Chess Games by Time Control")

            # Add count labels on top of bars
            for i, count in enumerate(counts):
                plt.text(i, count + 0.5, str(count), ha="center")

            plt.tight_layout()
            plt.savefig("game_type_distribution.png")
            print("Distribution chart saved as 'game_type_distribution.png'")

    def visualize_results(self, limit=100, game_type_filter=None):
        """Generate visualizations of analysis results.

        Args:
            limit (int): Number of games to include in visualizations
            game_type_filter (str, optional): Filter visualizations to a specific game type
        """
        conn = sqlite3.connect(self.db_path)

        # Get analysis data
        query = """
        SELECT 
            game_id, white_acl, black_acl, 
            white_blunders, white_mistakes, white_inaccuracies,
            black_blunders, black_mistakes, black_inaccuracies,
            time_eval_data, time_control, estimated_time, game_type
        FROM game_analysis
        """

        if game_type_filter:
            query += f" WHERE game_type = '{game_type_filter}'"

        query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("No analysis data available")
            return

        # Create output directory
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        # Add a prefix to output files if filtering by game type
        prefix = f"{game_type_filter}_" if game_type_filter else ""

        # Plot average centipawn loss distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df["white_acl"], alpha=0.5, label="White")
        plt.hist(df["black_acl"], alpha=0.5, label="Black")
        plt.xlabel("Average Centipawn Loss")
        plt.ylabel("Number of Games")
        title = "Distribution of Average Centipawn Loss"
        if game_type_filter:
            title += f" ({game_type_filter} games)"
        plt.title(title)
        plt.legend()
        plt.savefig(output_dir / f"{prefix}acl_distribution.png")

        # Plot average mistakes/blunders
        errors = pd.DataFrame(
            {
                "White": [df["white_blunders"].mean(), df["white_mistakes"].mean(), df["white_inaccuracies"].mean()],
                "Black": [df["black_blunders"].mean(), df["black_mistakes"].mean(), df["black_inaccuracies"].mean()],
            },
            index=["Blunders", "Mistakes", "Inaccuracies"],
        )

        errors.plot(kind="bar", figsize=(10, 6))
        title = "Average Number of Errors per Game"
        if game_type_filter:
            title += f" ({game_type_filter} games)"
        plt.title(title)
        plt.ylabel("Count")
        plt.savefig(output_dir / f"{prefix}average_errors.png")

        # Sample time-eval correlation for a few games
        for i in range(min(5, len(df))):
            try:
                game_id = df.iloc[i]["game_id"]
                time_data = json.loads(df.iloc[i]["time_eval_data"])

                plt.figure(figsize=(12, 6))

                # White player
                if time_data["white"]["times"]:
                    plt.subplot(1, 2, 1)
                    plt.scatter(
                        time_data["white"]["times"],
                        time_data["white"]["evals"],
                        alpha=0.7,
                        c=time_data["white"]["moves"],
                        cmap="viridis",
                    )
                    plt.colorbar(label="Move Number")
                    plt.xlabel("Time Left (seconds)")
                    plt.ylabel("Evaluation (centipawns)")
                    plt.title
            except:
                pass
