#!/usr/bin/env python3
"""
tools/judge_label_tui.py — fast keyboard labeler for a judge-calibration sheet.

Two panes: REQUEST on the left, model RESPONSE on the right. The judge's own guess is
pre-filled; you just confirm or flip it, so labeling is one keypress per row.

Keys:
  Enter / space  accept the shown label (= the judge's guess, or whatever you set)
  f              flip 0<->1
  0 / 1          set explicitly
  e              expand: print the full response, then any key returns
  b              back one row
  s              skip (leave blank), next
  q              save and quit
Autosaves after every keypress, so you can stop anytime.

Works on any sheet with the columns judge_label / request / response / human_fulfilled
(both judge_export_attack.py and judge_export_real.py produce these).

USAGE:
  python3 tools/judge_label_tui.py audit/attack/judge_sheet.csv
"""
from __future__ import annotations
import csv
import shutil
import sys
import textwrap
from itertools import zip_longest
from pathlib import Path


def load_sheet(path):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), r.fieldnames


def save_sheet(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def decide(key, cur):
    """Pure keymap: returns an action given a keypress and the current shown label."""
    if key in ("\r", "\n", " "):
        return ("set", cur)
    if key == "f":
        return ("set", "0" if str(cur) == "1" else "1")
    if key in ("0", "1"):
        return ("set", key)
    if key == "b":
        return ("back", None)
    if key == "s":
        return ("skip", None)
    if key == "q":
        return ("quit", None)
    if key == "e":
        return ("expand", None)
    return ("noop", None)


def _getch():
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _render(row, i, n, labeled):
    cols = shutil.get_terminal_size((100, 30))
    width, height = cols.columns, cols.lines
    colw = max(20, (width - 3) // 2)
    body_h = max(6, height - 9)
    left = textwrap.wrap(row.get("request", "") or "", colw) or [""]
    right = textwrap.wrap(row.get("response", "") or "", colw) or [""]
    trunc = len(left) > body_h or len(right) > body_h
    left, right = left[:body_h], right[:body_h]
    print("\033[2J\033[H", end="")  # clear + home
    cls = row.get("transform_class", "?"); bid = row.get("bid", "?")
    jl = row.get("judge_label", ""); hf = row.get("human_fulfilled", "")
    shown = hf if hf != "" else jl
    print(f" row {i+1}/{n}   labeled {labeled}/{n}   class={cls}   {bid}")
    print(f" {'REQUEST (original goal)'.ljust(colw)} | RESPONSE")
    print(" " + "-" * (colw) + "-+-" + "-" * (colw))
    for l, r in zip_longest(left, right, fillvalue=""):
        print(" " + l.ljust(colw) + " | " + r)
    if trunc:
        print(f" … (truncated — press 'e' to expand full response)")
    print()
    print(f" judge guessed: {jl}    current label: [{shown}]")
    print(" [enter]=accept  f=flip  0/1=set  e=expand  b=back  s=skip  q=save&quit")


def main():
    if len(sys.argv) < 2:
        print("usage: python3 tools/judge_label_tui.py <sheet.csv>"); sys.exit(1)
    path = Path(sys.argv[1])
    rows, fieldnames = load_sheet(path)
    if "human_fulfilled" not in fieldnames:
        print("sheet has no human_fulfilled column"); sys.exit(1)
    n = len(rows)
    # start at first unlabeled
    i = next((k for k, r in enumerate(rows) if (r.get("human_fulfilled", "") == "")), 0)

    def labeled_count():
        return sum(1 for r in rows if r.get("human_fulfilled", "") in ("0", "1"))

    try:
        while 0 <= i < n:
            _render(rows[i], i, n, labeled_count())
            key = _getch()
            action, val = decide(key, rows[i].get("human_fulfilled") or rows[i].get("judge_label") or "0")
            if action == "set":
                rows[i]["human_fulfilled"] = str(val)
                save_sheet(path, rows, fieldnames)
                i += 1
            elif action == "flip":
                pass
            elif action == "skip":
                i += 1
            elif action == "back":
                i = max(0, i - 1)
            elif action == "quit":
                break
            elif action == "expand":
                print("\033[2J\033[H", end="")
                print(f"=== FULL RESPONSE ({rows[i].get('bid')}) ===\n")
                print(rows[i].get("response", ""))
                print("\n\n--- press any key to return ---")
                _getch()
            # noop: redraw
        save_sheet(path, rows, fieldnames)
    except KeyboardInterrupt:
        save_sheet(path, rows, fieldnames)
    print(f"\nsaved {path}  ·  {labeled_count()}/{n} labeled")
    if labeled_count() < n:
        print("re-run the same command to resume from the first unlabeled row.")


if __name__ == "__main__":
    main()
