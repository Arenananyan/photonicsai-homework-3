"""Mac-side SSH transport and simulation. No hardware access at import."""
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from printer_policy import DEFAULT_Z_SPEED, SafetyError, check_execution_state, check_state, plan, plan_absolute, plan_xy_start, same_position, travel_limits
from printer_watchdog import AUDITED_HOMING_SIGNATURE, homing_state, validate_request


class PrinterError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def mock_state():
    pos = [100.0, 100.0, 5.0, 0.0]
    return {'webhooks': {'state': 'ready'}, 'print_stats': {'state': 'standby'},
            'toolhead': {'position': pos[:], 'homed_axes': 'xyz',
                         'axis_minimum': [-12., -20., -5., 0.], 'axis_maximum': [221., 223., 246., 0.]},
            'motion_report': {'live_position': pos[:], 'live_velocity': 0.0},
            'gcode_move': {'position': pos[:], 'gcode_position': pos[:],
                           'homing_origin': [0., 0., 0., 0.], 'speed_factor': 1.0},
            'heater_bed': {'target': 0.0}, 'extruder': {'target': 0.0},
            'bed_mesh': {'profile_name': '', 'mesh_matrix': [[]]},
            'homing_lift_mm': 5.0,
            'motion_limits': {'max_velocity': 500.0, 'max_z_velocity': 30.0,
                              'max_accel': 8000.0, 'max_z_accel': 300.0}}


class MockPrinter:
    simulated = True

    def __init__(self):
        self.state = mock_state()
        self.submitted_scripts = []

    def request(self, request):
        action = validate_request(request)
        if action == 'status':
            return {'ok': True, 'simulated': True, 'state': copy.deepcopy(self.state)}
        if action == 'capabilities':
            return {'ok': True, 'simulated': True, 'motion_limits': copy.deepcopy(self.state['motion_limits']),
                    'homing_configuration': {}, 'homing_macros': {}, 'steppers': {},
                    'homing_signature': AUDITED_HOMING_SIGNATURE}
        if action in ('home-preview', 'home'):
            axes = request['axes']
            if (not isinstance(axes, list) or not axes or any(not isinstance(a, str) or a not in ('x', 'y', 'z') for a in axes)
                    or len(set(axes)) != len(axes)):
                raise SafetyError('Homing axes must be a nonempty unique list of x, y, z')
            start = homing_state(self.state, stationary=True)
            prepared = {'axes': sorted(axes), 'start': start,
                        'homed_axes': self.state['toolhead']['homed_axes'],
                        'homing_signature': AUDITED_HOMING_SIGNATURE,
                        'message': 'Simulated firmware homing; real homing may move other axes and change mesh'}
            if action == 'home-preview':
                return {'ok': True, 'simulated': True, 'plan': prepared}
            if (not same_position(start, request['expected_start'])
                    or request['expected_homed_axes'] != prepared['homed_axes']
                    or request['homing_signature'] != AUDITED_HOMING_SIGNATURE):
                raise SafetyError('Homing state changed since preview')
            self.state['toolhead']['homed_axes'] = ''.join(sorted(set(prepared['homed_axes']) | set(axes)))
            self.submitted_scripts.append('G28 ' + ' '.join(a.upper() for a in sorted(axes)) + '\nM400')
            return {'ok': True, 'simulated': True, 'reported_position': start,
                    'homed_axes': self.state['toolhead']['homed_axes']}
        if action == 'stop':
            self.state['webhooks']['state'] = 'shutdown'
            return {'ok': True, 'stop_requested': True, 'stop_confirmed': True}
        if action in ('mesh-clear', 'mesh-load'):
            check_state(self.state)
            self.state['bed_mesh'] = {'profile_name': request.get('profile', ''),
                                      'mesh_matrix': [[0.0]] if action == 'mesh-load' else [[]]}
            return {'ok': True, 'simulated': True, 'bed_mesh': copy.deepcopy(self.state['bed_mesh'])}
        if 'expected_limits_mm' in request and request['expected_limits_mm'] != travel_limits(self.state):
            raise SafetyError('Firmware travel limits changed since preview; preview again')
        prepared = (plan_absolute(self.state, request['target'], request['speed_mm_s'])
                    if action in ('absolute-preview', 'absolute-move') else
                    plan_xy_start(self.state) if action in ('xy-start', 'xy-start-preview') else
                    plan(self.state, request['steps'], request['speed_mm_s'], request.get('z_speed_mm_s', DEFAULT_Z_SPEED)))
        if action in ('preview', 'absolute-preview', 'xy-start-preview'):
            return {'ok': True, 'simulated': True, 'plan': prepared}
        if not same_position(prepared['start'], request['expected_start']):
            raise SafetyError('Position changed since preview')
        if prepared['targets']:
            self.submitted_scripts.append(prepared['script'])
        for target in prepared['targets']:
            for obj, field in (('toolhead', 'position'), ('gcode_move', 'position'),
                               ('gcode_move', 'gcode_position'), ('motion_report', 'live_position')):
                self.state[obj][field][:3] = target
            check_execution_state(self.state, prepared)
        return {'ok': True, 'simulated': True, 'reported_position': prepared['end'],
                'commands': len(prepared['targets'])}


class SSHPrinter:
    simulated = False

    def __init__(self):
        self.host = os.getenv('SSH_HOST', '')
        self.control = os.getenv('SSH_CONTROL_PATH', '/tmp/photonics-printer-control')
        self.python = '/usr/share/klippy-env/bin/python'
        if not self.host:
            raise PrinterError('Set SSH_HOST to your printer IP in this terminal before using hardware.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', self.host):
            raise PrinterError('Invalid SSH_HOST')
        if not self.control.startswith('/') or any(ord(c) < 32 for c in self.control):
            raise PrinterError('SSH_CONTROL_PATH must be an absolute path without control characters')

    def arguments(self):
        return ['ssh', '-F', '/dev/null', '-S', self.control, '-o', 'BatchMode=yes',
                '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5']

    def source(self):
        root = Path(__file__).parent
        policy = (root / 'printer_policy.py').read_text()
        guard = (root / 'printer_watchdog.py').read_text()
        return ("import sys, types\n"
                "policy = types.ModuleType('printer_policy')\n"
                "exec(" + repr(policy) + ", policy.__dict__)\n"
                "sys.modules['printer_policy'] = policy\n"
                "exec(" + repr(guard) + ", {'__name__': '__main__'})\n")

    def request(self, request):
        action = validate_request(request)
        payload = json.dumps(request, allow_nan=False)
        if len(payload.encode()) > 65536:
            raise PrinterError('Request is too large')
        destination = 'root@' + self.host
        try:
            check = subprocess.run(self.arguments() + ['-O', 'check', destination],
                                   capture_output=True, text=True, timeout=5)
            if check.returncode:
                raise PrinterError('Shared SSH session unavailable. Run ./printerctl connect in another Mac Terminal tab.')
            # Embedded SSH servers may reject long exec commands and close the
            # shared connection. Send code as one JSON line on stdin, followed
            # by the request. Read both through the same buffered stream so
            # Python read-ahead cannot swallow the request.
            bootstrap = ("import json,sys; "
                         "exec(compile(json.loads(sys.stdin.buffer.readline(1048577)), "
                         "'<photonics-watchdog>', 'exec'), {'__name__': '__main__'})")
            remote = shlex.quote(self.python) + ' -c ' + shlex.quote(bootstrap)
            wire_input = json.dumps(self.source()) + '\n' + payload
            result = subprocess.run(self.arguments() + [destination, remote], input=wire_input,
                                    capture_output=True, text=True,
                                    timeout=315 if action in ('run', 'absolute-move', 'home', 'xy-start') else 10)
            if result.returncode:
                detail = result.stderr.strip()[:400]
                if not detail and result.returncode == 255:
                    detail = ('Connection closed while starting or running the printer helper (exit 255). '
                              'Check the connect tab for details, then reconnect with ./printerctl connect.')
                raise PrinterError('SSH failed: ' + (detail or 'exit %s' % result.returncode))
            reply = json.loads(result.stdout)
            if not isinstance(reply, dict) or not isinstance(reply.get('ok'), bool):
                raise PrinterError('Invalid watchdog response')
            return reply
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            raise PrinterError('SSH/watchdog response failed: ' + str(exc)) from exc


def checked(printer, request):
    reply = printer.request(request)
    if not reply.get('ok'):
        raise PrinterError(json.dumps(reply))
    return reply


def run_prepared(printer, steps, speed, prepared):
    """No movement retries. Attempt dedicated stop if the SSH outcome is lost."""
    request = ({'action': 'absolute-move', 'target': prepared['target'],
                'speed_mm_s': prepared['speed_mm_s'], 'expected_start': prepared['start']}
               if prepared.get('absolute') else
               {'action': 'xy-start', 'expected_start': prepared['start']}
               if prepared.get('xy_start') else
               {'action': 'run', 'steps': steps, 'speed_mm_s': speed,
                'z_speed_mm_s': prepared.get('z_speed_mm_s', DEFAULT_Z_SPEED),
                'expected_start': prepared['start']})
    if 'firmware_limits_mm' in prepared:
        request['expected_limits_mm'] = prepared['firmware_limits_mm']
    try:
        reply = printer.request(request)
    except (PrinterError, OSError, KeyboardInterrupt) as exc:
        # The on-printer watchdog is the primary monitor. This second stop path
        # helps when the Mac process loses its reply or the user presses Ctrl-C.
        try:
            stop = printer.request({'action': 'stop'})
            confirmed = stop.get('stop_confirmed', False)
        except Exception:
            confirmed = False
        raise PrinterError('Sequence outcome uncertain; stop requested (confirmed=%s). '
                           'Inspect the printer before another run. %s' % (confirmed, exc),
                           {'command_may_have_executed': True, 'stop_requested': True,
                            'stop_confirmed': confirmed}) from exc
    if not reply.get('ok'):
        # An explicit preflight rejection needs no stop; the remote watchdog
        # already requested stop for errors after dispatch and reports that here.
        raise PrinterError(json.dumps(reply))
    return reply
