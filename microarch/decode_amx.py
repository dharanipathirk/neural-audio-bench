#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
"""
AMX Instruction Decoder for Apple Silicon Binaries

Scans a running process or a Mach-O binary for Apple's undocumented AMX
(Apple Matrix Extensions) coprocessor instructions and decodes them.

AMX encoding reference: dougallj/aarch64_amx.py
https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f

Usage:
    # Scan a running process by PID
    python decode_amx.py --pid 12345

    # Scan a running process by name (attaches briefly via lldb)
    python decode_amx.py --process bench_bnns

    # Decode raw instruction bytes from stdin (one hex word per line)
    echo "0x0020100d" | python decode_amx.py --raw

    # Scan lldb disassembly output piped in
    lldb -p PID --batch -o "di -s ADDR -c 200 -b" | python decode_amx.py --lldb

    # Run a benchmark and scan its hot path automatically
    python decode_amx.py --launch "./build/bench_bnns models/conv1d_1m.mlmodelc"
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass

# AMX operation table (op field = bits 5-9)
AMX_OPS = {
    0: ("AMXLDX", "Load X matrix register"),
    1: ("AMXLDY", "Load Y matrix register"),
    2: ("AMXSTX", "Store X matrix register"),
    3: ("AMXSTY", "Store Y matrix register"),
    4: ("AMXLDZ", "Load Z accumulator register"),
    5: ("AMXSTZ", "Store Z accumulator register"),
    6: ("AMXLDZI", "Load Z interleaved"),
    7: ("AMXSTZI", "Store Z interleaved"),
    8: ("AMXEXTRX", "Extract row from X"),
    9: ("AMXEXTRY", "Extract row from Y"),
    10: ("AMXFMA64", "FP64 fused multiply-add"),
    11: ("AMXFMS64", "FP64 fused multiply-sub"),
    12: ("AMXFMA32", "FP32 fused multiply-add"),
    13: ("AMXFMS32", "FP32 fused multiply-sub"),
    14: ("AMXMAC16", "INT16 multiply-accumulate"),
    15: ("AMXFMA16", "FP16 fused multiply-add"),
    16: ("AMXFMS16", "FP16 fused multiply-sub"),
    17: ("AMX17", "AMX enable/disable"),
    18: ("AMXVECINT", "Vector integer op"),
    19: ("AMXVECFP", "Vector floating-point op"),
    20: ("AMXMATINT", "Matrix integer op"),
    21: ("AMXMATFP", "Matrix floating-point op"),
    22: ("AMXGENLUT", "Generate lookup table"),
}

AMX_MASK = 0xFFFFFC00
AMX_MATCH = 0x00201000


@dataclass
class AMXInstruction:
    address: int
    encoding: int
    op: int
    operand: int
    mnemonic: str
    description: str
    library: str = ""


def is_amx(insn: int) -> bool:
    """Check if a 32-bit instruction word is an AMX instruction."""
    return (insn & AMX_MASK) == AMX_MATCH


def decode_amx(insn: int, address: int = 0, library: str = "") -> AMXInstruction:
    """Decode a 32-bit AMX instruction word."""
    op = (insn >> 5) & 0x1F
    operand = insn & 0x1F
    mnemonic, description = AMX_OPS.get(op, (f"AMX_OP{op}", "Unknown AMX operation"))
    return AMXInstruction(
        address=address,
        encoding=insn,
        op=op,
        operand=operand,
        mnemonic=mnemonic,
        description=description,
        library=library,
    )


def parse_lldb_output(text: str) -> list[AMXInstruction]:
    """Parse lldb disassembly output with -b (bytes) flag and find AMX instructions."""
    results = []
    # Match lines like: 0x19d6c908c <+456>: 0x0020100d   <unknown>
    # Also match: 0x19d6c9070 <+428>: 0xf85b81ac   ldur   x12, [x29, #-0x68]
    pattern = re.compile(r"(0x[0-9a-f]+)\s+.*?:\s+0x([0-9a-f]{8})")

    current_lib = ""
    for line in text.splitlines():
        # Track which library we're in
        lib_match = re.match(r"(\S+\.dylib)`", line)
        if lib_match:
            current_lib = lib_match.group(1)

        m = pattern.search(line)
        if m:
            addr = int(m.group(1), 16)
            insn = int(m.group(2), 16)
            if is_amx(insn):
                results.append(decode_amx(insn, addr, current_lib))
    return results


def scan_process(pid: int, sample_addresses: list[int] | None = None) -> list[AMXInstruction]:
    """Attach to a process via lldb and scan for AMX instructions around hot addresses."""
    if not sample_addresses:
        # Default: scan the most common BNNSGraph hot regions
        # We'll ask lldb to disassemble a wide range
        print(f"Sampling process {pid} to find hot addresses...", file=sys.stderr)
        sample_result = subprocess.run(
            ["sample", str(pid), "2"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Extract addresses from libBNNS.dylib
        addresses = []
        for line in sample_result.stdout.splitlines():
            m = re.search(r"(0x[0-9a-f]+)\s+.*libBNNS", line)
            if m:
                addresses.append(int(m.group(1), 16))

        if not addresses:
            # Try libBLAS too (LibTorch path)
            for line in sample_result.stdout.splitlines():
                m = re.search(r"(0x[0-9a-f]+)\s+.*libBLAS", line)
                if m:
                    addresses.append(int(m.group(1), 16))

        if not addresses:
            print(
                "No BNNS/BLAS addresses found in sample. Process may not be running inference.",
                file=sys.stderr,
            )
            return []

        # Find the hottest address region
        counter = Counter(a & ~0xFFF for a in addresses)  # Page-align
        hottest_page = counter.most_common(1)[0][0]
        sample_addresses = [hottest_page]

    # Disassemble around hot addresses
    all_results = []
    for base_addr in sample_addresses:
        lldb_commands = [
            f"di -s {hex(base_addr)} -c 256 -b",
        ]
        cmd_args = ["lldb", "-p", str(pid), "--batch"]
        for c in lldb_commands:
            cmd_args.extend(["-o", c])

        result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=15)
        all_results.extend(parse_lldb_output(result.stdout + result.stderr))

    return all_results


def launch_and_scan(command: str) -> list[AMXInstruction]:
    """Launch a process, let it run briefly, scan for AMX, then kill it."""
    import shlex

    args = shlex.split(command)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Launched PID {proc.pid}: {command}", file=sys.stderr)
    time.sleep(3)  # Let it warm up and start inference

    results = scan_process(proc.pid)

    proc.kill()
    proc.wait()
    return results


def print_results(instructions: list[AMXInstruction], verbose: bool = False):
    """Print decoded AMX instructions and summary statistics."""
    if not instructions:
        print("No AMX instructions found.")
        return

    # Group by library
    by_library: dict[str, list[AMXInstruction]] = {}
    for insn in instructions:
        lib = insn.library or "(unknown)"
        by_library.setdefault(lib, []).append(insn)

    for lib, insns in sorted(by_library.items()):
        print(f"\n{'=' * 70}")
        print(f"Library: {lib}")
        print(f"{'=' * 70}")

        if verbose:
            print(f"\n{'Address':<18} {'Encoding':<12} {'Mnemonic':<14} {'Reg':<6} {'Description'}")
            print("-" * 70)
            for insn in insns:
                print(
                    f"0x{insn.address:016x} 0x{insn.encoding:08x}   "
                    f"{insn.mnemonic:<14} x{insn.operand:<4} {insn.description}"
                )

        # Summary
        op_counts = Counter(insn.mnemonic for insn in insns)
        print(f"\nInstruction mix ({len(insns)} AMX instructions):")
        for name, count in op_counts.most_common():
            desc = next((d for m, d in AMX_OPS.values() if m == name), "")
            bar = "#" * count
            print(f"  {name:<14} {count:>3}  {bar}  {desc}")

        # Categorize
        loads = sum(c for n, c in op_counts.items() if "LD" in n)
        stores = sum(c for n, c in op_counts.items() if "ST" in n)
        compute = sum(
            c
            for n, c in op_counts.items()
            if any(k in n for k in ["FMA", "FMS", "MAC", "MAT", "VEC"])
        )
        extract = sum(c for n, c in op_counts.items() if "EXTR" in n)
        control = sum(c for n, c in op_counts.items() if n == "AMX17")

        print("\nCategory breakdown:")
        print(f"  Loads:   {loads:>3}  (data → AMX registers)")
        print(f"  Compute: {compute:>3}  (FMA/matrix/vector operations)")
        print(f"  Stores:  {stores:>3}  (AMX registers → memory)")
        print(f"  Extract: {extract:>3}  (row/column extraction)")
        print(f"  Control: {control:>3}  (enable/disable)")

        # Precision analysis
        # Match FP operations only (FMA/FMS), not integer ops like AMXMAC16.
        fp32 = sum(c for n, c in op_counts.items() if "32" in n and ("FMA" in n or "FMS" in n))
        fp16 = sum(c for n, c in op_counts.items() if "16" in n and ("FMA" in n or "FMS" in n))
        fp64 = sum(c for n, c in op_counts.items() if "64" in n and ("FMA" in n or "FMS" in n))
        int16 = sum(c for n, c in op_counts.items() if "MAC16" in n)
        if fp32 or fp16 or fp64 or int16:
            print("\nPrecision:")
            if fp32:
                print(f"  FP32:  {fp32} operations")
            if fp16:
                print(f"  FP16:  {fp16} operations")
            if fp64:
                print(f"  FP64:  {fp64} operations")
            if int16:
                print(f"  INT16: {int16} operations")

        # Compute intensity
        if loads > 0 and compute > 0:
            print(f"\nCompute intensity: {compute / loads:.2f} compute ops per load")
            print("  (Higher = better AMX utilization, weight reuse)")


def build_json_output(
    instructions: list[AMXInstruction],
    backend: str,
    pid: int | None,
) -> dict:
    """Build the JSON output structure from decoded AMX instructions."""
    libraries: dict[str, dict] = {}

    # Group by library (mirrors print_results logic)
    by_library: dict[str, list[AMXInstruction]] = {}
    for insn in instructions:
        lib = insn.library or "(unknown)"
        by_library.setdefault(lib, []).append(insn)

    for lib, insns in sorted(by_library.items()):
        op_counts = Counter(insn.mnemonic for insn in insns)

        loads = sum(c for n, c in op_counts.items() if "LD" in n)
        stores = sum(c for n, c in op_counts.items() if "ST" in n)
        compute = sum(
            c
            for n, c in op_counts.items()
            if any(k in n for k in ["FMA", "FMS", "MAC", "MAT", "VEC"])
        )
        extract = sum(c for n, c in op_counts.items() if "EXTR" in n)
        control = sum(c for n, c in op_counts.items() if n == "AMX17")

        fp32 = sum(c for n, c in op_counts.items() if "32" in n and ("FMA" in n or "FMS" in n))
        fp16 = sum(c for n, c in op_counts.items() if "16" in n and ("FMA" in n or "FMS" in n))
        fp64 = sum(c for n, c in op_counts.items() if "64" in n and ("FMA" in n or "FMS" in n))
        int16 = sum(c for n, c in op_counts.items() if "MAC16" in n)

        compute_intensity = (compute / loads) if loads > 0 else 0.0

        libraries[lib] = {
            "total_insns": len(insns),
            "loads": loads,
            "compute": compute,
            "stores": stores,
            "extract": extract,
            "control": control,
            "fp32": fp32,
            "fp16": fp16,
            "fp64": fp64,
            "int16": int16,
            "compute_intensity": round(compute_intensity, 4),
            "op_counts": dict(op_counts.most_common()),
        }

    return {
        "backend": backend,
        "timestamp": datetime.datetime.now().isoformat(),
        "pid": pid,
        "libraries": libraries,
    }


def infer_backend(args: argparse.Namespace) -> str:
    """Infer backend name from CLI arguments when --backend is not provided."""
    if hasattr(args, "process") and args.process:
        return args.process
    if hasattr(args, "launch") and args.launch:
        # Use the basename of the first token of the launch command
        import shlex

        tokens = shlex.split(args.launch)
        if tokens:
            import os

            return os.path.basename(tokens[0])
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Decode Apple AMX instructions in running processes or binaries",
        allow_abbrev=False,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pid", type=int, help="Attach to running process by PID")
    group.add_argument("--process", type=str, help="Process name to find and attach to")
    group.add_argument("--launch", type=str, help="Command to launch, scan, and kill")
    group.add_argument("--lldb", action="store_true", help="Parse lldb disassembly from stdin")
    group.add_argument("--raw", action="store_true", help="Decode hex instruction words from stdin")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show individual instructions")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend name to embed in JSON output (e.g. BNNSGraph). "
        "Inferred from --process/--launch when omitted.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Write structured JSON results to this file path after printing human-readable output.",
    )

    args = parser.parse_args()

    pid_used: int | None = args.pid if hasattr(args, "pid") else None

    if args.raw:
        instructions = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                insn = int(line, 16)
                if is_amx(insn):
                    instructions.append(decode_amx(insn))
            except ValueError:
                pass
        print_results(instructions, verbose=True)

    elif args.lldb:
        text = sys.stdin.read()
        instructions = parse_lldb_output(text)
        print_results(instructions, args.verbose)

    elif args.launch:
        instructions = launch_and_scan(args.launch)
        print_results(instructions, args.verbose)

    elif args.pid:
        pid_used = args.pid
        instructions = scan_process(args.pid)
        print_results(instructions, args.verbose)

    elif args.process:
        result = subprocess.run(["pgrep", "-f", args.process], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        if not pids or not pids[0]:
            print(f"No process found matching '{args.process}'", file=sys.stderr)
            sys.exit(1)
        pid_used = int(pids[0])
        print(f"Found PID {pid_used} for '{args.process}'", file=sys.stderr)
        instructions = scan_process(pid_used)
        print_results(instructions, args.verbose)

    if args.json_output:
        backend = args.backend if args.backend else infer_backend(args)
        payload = build_json_output(instructions, backend=backend, pid=pid_used)
        import os

        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"JSON results written to {args.json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
