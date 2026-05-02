"""
Warning Logger for EcoGuard AI Platform.
Logs every detection session and its warnings to a JSON file,
so they persist and can be displayed as charts on the dashboard.
"""

import json
import os
import threading
from datetime import datetime
from django.conf import settings

# Path to the persistent log file
LOG_FILE = os.path.join(settings.BASE_DIR, 'detection_log.json')
_lock = threading.Lock()


def _read_log():
    """Read the current log file, return list of entries."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _write_log(entries):
    """Write entries list back to log file."""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, default=str)


def log_detection(module, detections_count, warnings_count, details=None):
    """
    Log a detection event.

    Args:
        module: str - which module was used (ppe, fish, smoke, animaux, illegal_mining, deforestation, mining)
        detections_count: int - total number of objects detected
        warnings_count: int - how many are warnings/violations
        details: dict - optional extra info (e.g. safe_count, unsafe_count, class breakdown)
    """
    entry = {
        'timestamp': datetime.now().isoformat(),
        'module': module,
        'detections': detections_count,
        'warnings': warnings_count,
        'details': details or {},
    }

    with _lock:
        entries = _read_log()
        entries.append(entry)
        # Keep only last 500 entries to avoid unbounded growth
        if len(entries) > 500:
            entries = entries[-500:]
        _write_log(entries)

    return entry


def get_all_logs():
    """Return all log entries."""
    with _lock:
        return _read_log()


def get_summary():
    """
    Build a summary for charts:
    - per-module total detections & warnings
    - timeline of warnings (last 30 sessions)
    - class breakdown
    """
    entries = get_all_logs()

    # Per-module totals
    module_stats = {}
    for e in entries:
        mod = e.get('module', 'unknown')
        if mod not in module_stats:
            module_stats[mod] = {'detections': 0, 'warnings': 0, 'sessions': 0}
        module_stats[mod]['detections'] += e.get('detections', 0)
        module_stats[mod]['warnings'] += e.get('warnings', 0)
        module_stats[mod]['sessions'] += 1

    # Timeline (last 30 entries)
    recent = entries[-30:] if len(entries) > 30 else entries
    timeline = []
    for e in recent:
        timeline.append({
            'timestamp': e.get('timestamp', ''),
            'module': e.get('module', ''),
            'detections': e.get('detections', 0),
            'warnings': e.get('warnings', 0),
        })

    # Total warnings vs safe
    total_warnings = sum(e.get('warnings', 0) for e in entries)
    total_detections = sum(e.get('detections', 0) for e in entries)
    total_sessions = len(entries)

    return {
        'module_stats': module_stats,
        'timeline': timeline,
        'total_warnings': total_warnings,
        'total_detections': total_detections,
        'total_sessions': total_sessions,
        'total_safe': total_detections - total_warnings,
    }
