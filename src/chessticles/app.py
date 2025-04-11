import streamlit as st
import chess
import chess.pgn
import chess.svg
import cairosvg
from io import BytesIO


# Set page config
st.set_page_config(page_title="Chess Game Viewer", layout="wide")

# Initialize session state variables
if "game" not in st.session_state:
    try:
        with open("data/data.pgn") as pgn:
            st.session_state.game = chess.pgn.read_game(pgn)
    except:
        st.error("Could not load PGN file")
        st.stop()

if "current_node" not in st.session_state:
    st.session_state.current_node = st.session_state.game

if "move_number" not in st.session_state:
    st.session_state.move_number = 0


def reset_game():
    st.session_state.current_node = st.session_state.game
    st.session_state.move_number = 0


def next_move():
    if st.session_state.current_node.variations:
        st.session_state.current_node = st.session_state.current_node.variations[0]
        st.session_state.move_number += 1


def prev_move():
    if st.session_state.current_node.parent:
        st.session_state.current_node = st.session_state.current_node.parent
        st.session_state.move_number -= 1


def go_to_end():
    end_node = st.session_state.game.end()
    st.session_state.current_node = end_node
    st.session_state.move_number = end_node.ply()


def svg_to_image(svg):
    """Convert SVG to high-resolution PNG."""
    png = cairosvg.svg2png(
        bytestring=svg,
        output_width=600,
        output_height=600,
        dpi=300,
    )
    return BytesIO(png)


# App title and header
st.title("Chess Game Viewer")

# Game info columns
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Game Information")
    st.write(f"**Event:** {st.session_state.game.headers.get('Event', 'Unknown')}")
    st.write(f"**White:** {st.session_state.game.headers.get('White', 'Unknown')}")
    st.write(f"**Black:** {st.session_state.game.headers.get('Black', 'Unknown')}")
    st.write(f"**Result:** {st.session_state.game.headers.get('Result', 'Unknown')}")

# Main content columns
chess_col, moves_col = st.columns([2, 1])

with chess_col:
    # Chessboard display
    st.subheader("Chessboard")
    board = st.session_state.current_node.board()
    svg = chess.svg.board(
        board,
        size=600,
        coordinates=True,
        lastmove=st.session_state.current_node.move if st.session_state.current_node.move else None,
        colors={"square light": "#F0D9B5", "square dark": "#B58863"},
    )
    image = svg_to_image(svg)
    st.image(image, use_container_width=False, caption="Current Position")

    # Move information
    current_ply = st.session_state.move_number
    total_ply = st.session_state.game.end().ply()
    st.subheader(f"Move {current_ply} of {total_ply}")
    if current_ply > 0:
        move = st.session_state.current_node.move
        parent_board = st.session_state.current_node.parent.board()
        san = parent_board.san(move)
        move_number = parent_board.fullmove_number
        prefix = f"{move_number}." if parent_board.turn == chess.WHITE else f"{move_number}..."
        st.write(f"**Current Move:** {prefix} {san}")

with moves_col:
    # Generate moves list
    moves = []
    node = st.session_state.game
    board = node.board()

    while node.variations:
        node = node.variation(0)
        move_number = board.fullmove_number
        prefix = f"{move_number}." if board.turn == chess.WHITE else "..."
        moves.append(f"{prefix} {board.san(node.move)}")
        board.push(node.move)

    white_moves = [m for i, m in enumerate(moves) if i % 2 == 0]
    black_moves = [m for i, m in enumerate(moves) if i % 2 == 1]

    # Create scrollable move list with buttons below
    total_ply = st.session_state.game.end().ply()
    current_ply = st.session_state.move_number

    scrollable_html = """
    <div style="height: 540px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between;">
            <div style="width: 48%;">
                <strong>White</strong><br>
    """

    # Add white moves
    for move in white_moves:
        scrollable_html += f"{move}<br>"

    scrollable_html += """
            </div>
            <div style="width: 48%;">
                <strong>Black</strong><br>
    """

    # Add black moves
    for i, move in enumerate(black_moves):
        san = move.split("... ")[1] if "... " in move else move
        move_number = white_moves[i].split(".")[0]
        scrollable_html += f"{move_number}... {san}<br>"

    scrollable_html += """
            </div>
        </div>
    </div>
    """

    st.markdown(scrollable_html, unsafe_allow_html=True)

    # Navigation buttons in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("⏮️", on_click=reset_game, disabled=current_ply == 0, help="First move")
    with col2:
        st.button("⏪", on_click=prev_move, disabled=current_ply == 0, help="Previous move")
    with col3:
        st.button("⏩", on_click=next_move, disabled=current_ply >= total_ply, help="Next move")
    with col4:
        st.button("⏭️", on_click=go_to_end, disabled=current_ply >= total_ply, help="Last move")
