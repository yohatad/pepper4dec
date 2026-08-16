"""conversation_manager_utilities.py

Terminal-color helpers and verbose-mode debug print formatting used by
conversation_manager_implementation.py.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0
"""

import rclpy.logging
from typing import Dict, List

logger = rclpy.logging.get_logger('conversation_manager')


# =============================================================================
# Terminal colours
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""

    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_CYAN = '\033[46m'
    BG_BLUE = '\033[44m'


# =============================================================================
# Verbose-mode printers
# Each function is a no-op when verbose=False.
# =============================================================================

def print_separator(verbose: bool, char: str = '=', length: int = 80,
                    color: str = Colors.BRIGHT_BLACK) -> None:
    """Log a colored separator line at INFO level when verbose."""
    if verbose:
        logger.info(f"{color}{char * length}{Colors.RESET}")


def print_message_header(verbose: bool, role: str, index: int = None) -> None:
    """Print a formatted message header."""
    if not verbose:
        return

    role_colors = {
        'system': Colors.BRIGHT_MAGENTA,
        'user': Colors.BRIGHT_CYAN,
        'assistant': Colors.BRIGHT_GREEN,
    }

    color = role_colors.get(role, Colors.WHITE)
    role_upper = role.upper()

    if index is not None:
        header = f"{color}{Colors.BOLD}[{role_upper} #{index}]{Colors.RESET}"
    else:
        header = f"{color}{Colors.BOLD}[{role_upper}]{Colors.RESET}"

    logger.info(header)


def print_message_content(verbose: bool, content: str, indent: int = 2) -> None:
    """Print message content with indentation and wrapping."""
    if not verbose:
        return

    indent_str = " " * indent
    lines = content.split('\n')
    logger.info('\n'.join(
        f"{Colors.BRIGHT_WHITE}{indent_str}{line}{Colors.RESET}" for line in lines
    ))


def print_search_results(verbose: bool, search_results: List[Dict]) -> None:
    """Print search results in a readable format."""
    if not verbose:
        return

    lines = [
        f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}"
        f"SEARCH RESULTS ({len(search_results)} documents){Colors.RESET}"
    ]
    for i, result in enumerate(search_results, 1):
        score = result.get('score', 0)
        score_color = (
            Colors.BRIGHT_GREEN if score > 0.5
            else Colors.YELLOW if score > 0.3
            else Colors.RED
        )
        content = result.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(
            f"{Colors.BRIGHT_CYAN}Result #{i}{Colors.RESET}  "
            f"{Colors.DIM}Doc:{Colors.RESET} {result.get('doc_id', 'N/A')}  "
            f"{Colors.DIM}Title:{Colors.RESET} {result.get('title', 'N/A')}  "
            f"{Colors.DIM}Score:{Colors.RESET} {score_color}{score:.4f}{Colors.RESET}"
        )
        for line in content.split('\n')[:3]:
            lines.append(f"    {Colors.BRIGHT_BLACK}{line}{Colors.RESET}")
    logger.info('\n'.join(lines))


def print_conversation_history(verbose: bool, conversation_history: List[Dict],
                               context_turns: int) -> None:
    """Print conversation history in a readable format."""
    if not verbose or not conversation_history:
        return

    history_to_use = conversation_history[-context_turns:]
    lines = [
        f"{Colors.BRIGHT_BLUE}{Colors.BOLD}CONVERSATION HISTORY "
        f"({len(history_to_use)}/{len(conversation_history)} turns used){Colors.RESET}"
    ]
    for i, turn in enumerate(history_to_use, 1):
        query = turn.get('query', '')
        response = turn.get('response', '')
        if len(response) > 150:
            response = response[:150] + "..."
        lines.append(
            f"{Colors.CYAN}Turn #{i}{Colors.RESET}  "
            f"{Colors.DIM}Q:{Colors.RESET} {Colors.BRIGHT_CYAN}{query}{Colors.RESET}  "
            f"{Colors.DIM}A:{Colors.RESET} {Colors.BRIGHT_GREEN}{response}{Colors.RESET}"
        )
    logger.info('\n'.join(lines))


def print_llm_request(verbose: bool, messages: List[Dict], model: str) -> None:
    """Print the complete LLM request in a readable format."""
    if not verbose:
        return

    lines = [
        f"{Colors.BG_BLUE}{Colors.BRIGHT_WHITE}{Colors.BOLD} LLM REQUEST {Colors.RESET}  "
        f"{Colors.DIM}Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{model}{Colors.RESET}  "
        f"{Colors.DIM}Messages:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(messages)}{Colors.RESET}"
    ]
    for i, message in enumerate(messages, 1):
        role = message.get('role', 'unknown').upper()
        content = message.get('content', '')
        lines.append(
            f"{Colors.BOLD}[{role} #{i}]{Colors.RESET} {content[:200]}"
            + ("..." if len(content) > 200 else "")
        )
    logger.info('\n'.join(lines))
