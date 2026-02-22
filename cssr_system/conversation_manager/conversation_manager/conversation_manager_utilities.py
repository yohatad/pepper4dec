"""
Utility helpers for the conversation manager.

Includes:
    - Colors: ANSI terminal colour codes
    - print_*: verbose-mode debug printers (each requires an explicit verbose: bool)
    - safe_*: type-coercion helpers used when reading YAML configuration
    - read_yaml_config: YAML file reader
"""

import yaml
import rclpy.logging
from typing import Any, Dict, List, Optional, Tuple

logger = rclpy.logging.get_logger('conversation_manager')


# =============================================================================
# Terminal colours
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'

    # Regular colors
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'

    # Bright colors
    BRIGHT_BLACK   = '\033[90m'
    BRIGHT_RED     = '\033[91m'
    BRIGHT_GREEN   = '\033[92m'
    BRIGHT_YELLOW  = '\033[93m'
    BRIGHT_BLUE    = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN    = '\033[96m'
    BRIGHT_WHITE   = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_CYAN  = '\033[46m'
    BG_BLUE  = '\033[44m'


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
        'system':    Colors.BRIGHT_MAGENTA,
        'user':      Colors.BRIGHT_CYAN,
        'assistant': Colors.BRIGHT_GREEN,
    }

    color     = role_colors.get(role, Colors.WHITE)
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
        score       = result.get('score', 0)
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
        query    = turn.get('query', '')
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
        role    = message.get('role', 'unknown').upper()
        content = message.get('content', '')
        lines.append(
            f"{Colors.BOLD}[{role} #{i}]{Colors.RESET} {content[:200]}"
            + ("..." if len(content) > 200 else "")
        )
    logger.info('\n'.join(lines))


# =============================================================================
# YAML / type-coercion helpers
# =============================================================================

def read_yaml_config(file_path: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Read configuration from a YAML file.

    Returns:
        Tuple of (config dict, error message or None)
    """
    try:
        logger.debug(f"Reading YAML config from: {file_path}")
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}, "YAML file is empty"
            if not isinstance(config, dict):
                return {}, f"YAML root must be a dictionary, got {type(config).__name__}"
            return config, None
    except FileNotFoundError:
        return {}, f"Config file not found: {file_path}"
    except yaml.YAMLError as e:
        return {}, f"Invalid YAML syntax: {e}"
    except Exception as e:
        return {}, f"Failed to read config file: {e}"


def safe_float(value: Any, default: float, name: str) -> Tuple[float, Optional[str]]:
    """Safely convert value to float."""
    if value is None:
        return default, None
    try:
        return float(value), None
    except (ValueError, TypeError):
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"


def safe_int(value: Any, default: int, name: str) -> Tuple[int, Optional[str]]:
    """Safely convert value to int."""
    if value is None:
        return default, None
    try:
        return int(value), None
    except (ValueError, TypeError):
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"


def safe_str(value: Any, default: str, name: str) -> Tuple[str, Optional[str]]:
    """Safely convert value to string."""
    if value is None:
        return default, None
    try:
        return str(value), None
    except Exception:
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"


def safe_bool(value: Any, default: bool, name: str) -> Tuple[bool, Optional[str]]:
    """Safely convert value to boolean."""
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True, None
        if value.lower() in ('false', 'no', '0', 'off'):
            return False, None
    try:
        return bool(value), None
    except Exception:
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"
