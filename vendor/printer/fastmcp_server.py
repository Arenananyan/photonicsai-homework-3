"""FastMCP 4 server. Defaults to simulation; returns dictionaries.

Real mode requires PRINTER_MODE=real. Execution additionally requires the
operator to set PRINTER_ALLOW_MOTION=1 before starting this process.
"""
import asyncio
import copy
import os
import threading
import time
import uuid

from fastmcp import FastMCP
from pydantic import StrictFloat
from printer_client import MockPrinter, SSHPrinter, checked, run_prepared
from printer_policy import DEFAULT_Z_SPEED
from printer_watchdog import STARTUP_ZERO_WINDOW
from printer_commands import PrinterCommands

mcp = FastMCP('Photonics Simple Printer')
mode = os.getenv('PRINTER_MODE', 'mock')
if mode not in ('mock', 'real'):
    raise ValueError('PRINTER_MODE must be mock or real')
printer = MockPrinter() if mode == 'mock' else SSHPrinter()
pending = {}
pending_lock = threading.Lock()
motion_lock = threading.Lock()


def invalidate_legacy_previews():
    with pending_lock:
        pending.clear()


commands = PrinterCommands(lambda: printer, motion_lock=motion_lock, on_invalidate=invalidate_legacy_previews)


def preview_impl(steps, speed_mm_s, z_speed_mm_s=DEFAULT_Z_SPEED):
    prepared = checked(printer, {'action': 'preview', 'steps': steps, 'speed_mm_s': speed_mm_s,
                                 'z_speed_mm_s': z_speed_mm_s})['plan']
    return remember_preview(steps, speed_mm_s, prepared)


def position_preview_impl(x, y, z, speed_mm_s):
    prepared = checked(printer, {'action': 'absolute-preview', 'target': {'x': x, 'y': y, 'z': z},
                                 'speed_mm_s': speed_mm_s})['plan']
    return remember_preview([], speed_mm_s, prepared)


def remember_preview(steps, speed_mm_s, prepared):
    token = uuid.uuid4().hex
    now = time.monotonic()
    with pending_lock:
        for key in list(pending):
            if pending[key]['expires'] < now:
                del pending[key]
        if len(pending) >= 100:
            raise ValueError('Too many pending previews')
        pending[token] = {'steps': copy.deepcopy(steps), 'speed': speed_mm_s,
                          'plan': prepared, 'expires': now + 120}
    return {'ok': True, 'simulated': printer.simulated, 'token': token, 'expires_in_seconds': 120,
            'absolute': prepared.get('absolute', False), 'end': prepared['end'],
            'start': prepared['start'], 'targets': prepared['targets'], 'bounds': prepared['bounds'],
            'z_bounds': prepared['z_bounds'], 'z_speed_mm_s': prepared['z_speed_mm_s'],
            'step_speeds_mm_s': prepared['step_speeds_mm_s'],
            'startup_zero_window_seconds': STARTUP_ZERO_WINDOW,
            'speed_mm_s': prepared['speed_mm_s'], 'nominal_seconds': prepared['nominal_seconds']}


def execute_impl(token, operator_confirmed):
    if operator_confirmed is not True:
        raise ValueError('Operator confirmation of the reviewed path is required')
    if not printer.simulated and os.getenv('PRINTER_ALLOW_MOTION') != '1':
        raise ValueError('Real MCP motion disabled; operator must set PRINTER_ALLOW_MOTION=1 locally')
    if not motion_lock.acquire(blocking=False):
        raise ValueError('A sequence is already executing')
    try:
        with pending_lock:
            prepared = pending.pop(token, None)
        if prepared is None or prepared['expires'] < time.monotonic():
            raise ValueError('Preview token is missing, expired, or already used')
        return run_prepared(printer, prepared['steps'], prepared['speed'], prepared['plan'])
    finally:
        motion_lock.release()


@mcp.tool
async def printer_command(request: dict) -> dict:
    """Version 1 JSON control. See docs/LLM_INTERFACE.md.

    Move: {command: move, target: {x: 100, y: 120, z: 30}, speed_mm_s: 25,
           preview: true}. This returns a single-use token. Execute the SAME target
    with preview: false, token, operator_confirmed: true after operator approval.
    Returns accepted/job_id; poll {command: job, job_id: ...} until completion.
    Other commands: status, position, capabilities, stop, home (same preview flow).
    Absolute mm; speed mm/s; maximum 30, further capped by firmware. Default 20.
    No heating/extrusion. Busy movements are rejected, never queued; reads/stop work.
    Real motion is locally gated; homing defaults to the touchscreen.
    """
    return await asyncio.to_thread(commands.execute, request)


@mcp.tool
async def printer_status() -> dict:
    """Read printer state, estimated live position and mesh; never moves."""
    return await asyncio.to_thread(checked, printer, {'action': 'status'})


@mcp.tool
async def preview_sequence(steps: list[dict], speed_mm_s: float = 20.0, z_speed_mm_s: float = DEFAULT_Z_SPEED) -> dict:
    """Validate all XYZ steps and return a 120-second, single-use preview token.

    Step example: {"axis": "x", "distance_mm": 50}. Negative distance moves back.
    Axes x/y/z accept scalar distances; xy accepts [dx,dy] for one straight XY leg. Travel limits are read from the printer, including negative coordinates.
    X/Y speed <=30 mm/s; Z speed <=30 mm/s, also capped by firmware. No heaters or homing.
    Live XYZ=0 with zero velocity may be missing telemetry for up to 0.5s at startup.
    """
    return await asyncio.to_thread(preview_impl, steps, speed_mm_s, z_speed_mm_s)


@mcp.tool
async def preview_position(x: StrictFloat, y: StrictFloat, z: StrictFloat,
                           speed_mm_s: StrictFloat = 20.0) -> dict:
    """Preview a straight move to absolute XYZ millimetres in the configured origin.

    Requires XYZ homed, zero coordinate offsets and inactive bed mesh.
    Bounds are the printer-reported firmware minimum/maximum for each axis.
    Read status/capabilities for current limits; no hardcoded travel range.
    Speed: >0 and <=30 mm/s, also capped by firmware. No heating or extrusion.
    Approve the whole physical path, then use execute_sequence with the token.
    """
    return await asyncio.to_thread(position_preview_impl, x, y, z, speed_mm_s)


@mcp.tool
async def execute_sequence(token: str, operator_confirmed: bool = False) -> dict:
    """Execute a reviewed sequence or absolute move once after operator approval.

    Uses on-printer status monitoring and automatic emergency-stop requests.
    Do not retry failed/uncertain movement without inspecting the hardware.
    """
    return await asyncio.to_thread(execute_impl, token, operator_confirmed)


@mcp.tool
async def emergency_stop() -> dict:
    """Dedicated emergency shutdown, independent of the movement lock; manual recovery required."""
    commands.invalidate()
    return await asyncio.to_thread(checked, printer, {'action': 'stop'})


if __name__ == '__main__':
    mcp.run(transport='stdio')
