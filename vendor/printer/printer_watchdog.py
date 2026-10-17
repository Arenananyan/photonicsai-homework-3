"""Runs beside Klipper via SSH; no firmware installation. Python 3.8 compatible."""
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import sys
import time

from printer_policy import DEFAULT_Z_SPEED, REPORT_ROUNDOFF, SafetyError, check_execution_state, check_state, firmware_speed_limits, mesh_active, number, plan, plan_absolute, plan_xy_start, same_position, startup_zero_report, travel_limits, xyz

SOCKET_PATH = '/tmp/klippy_uds'
LOCK_PATH = '/tmp/photonics-simple-motion.lock'
POLL_INTERVAL = 0.10
RESPONSE_TIMEOUT = 1.0
STARTUP_ZERO_WINDOW = 0.5
MAX_BYTES = 1048576
# Inspected Ender-3 V3 KE homing macros; includes a 5 mm pre-homing Z lift.
AUDITED_HOMING_SIGNATURE = 'f823ecee4bd4f6cdec9fa36fb37900a948d359a98a43a8364b0214bc4b069aa1'


class API:
    def __init__(self, path=SOCKET_PATH):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(RESPONSE_TIMEOUT)
        try:
            self.sock.connect(path)
        except BaseException:
            self.sock.close()
            raise
        self.buffer, self.replies, self.counter = b'', {}, 0

    def close(self):
        self.sock.close()

    def send(self, method, params=None):
        self.counter += 1
        message = {'id': self.counter, 'method': method, 'params': params or {}}
        self.sock.sendall(json.dumps(message, allow_nan=False).encode() + b'\x03')
        return self.counter

    def result(self, ident):
        reply = self.replies.pop(ident)
        if 'error' in reply:
            raise RuntimeError('Klipper rejected request: ' + str(reply['error']))
        if not isinstance(reply.get('result'), dict):
            raise RuntimeError('Invalid Klipper response')
        return reply['result']

    def wait(self, ident, timeout=RESPONSE_TIMEOUT):
        deadline = time.monotonic() + timeout
        while ident not in self.replies:
            while b'\x03' not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Klipper status/response timeout')
                self.sock.settimeout(remaining)
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise RuntimeError('Klipper socket closed')
                self.buffer += chunk
                if len(self.buffer) > MAX_BYTES:
                    raise RuntimeError('Oversized Klipper response')
            frame, self.buffer = self.buffer.split(b'\x03', 1)
            reply = json.loads(frame)
            if not isinstance(reply, dict) or not isinstance(reply.get('id'), int):
                raise RuntimeError('Invalid Klipper response envelope')
            self.replies[reply['id']] = reply
            if len(self.replies) > 16:
                raise RuntimeError('Unexpected Klipper response backlog')
        return self.result(ident)

    def call(self, method, params=None, timeout=RESPONSE_TIMEOUT):
        deadline = time.monotonic() + timeout
        self.sock.settimeout(timeout)
        ident = self.send(method, params)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Klipper status/response timeout')
        return self.wait(ident, remaining)


def query_objects(api):
    available = api.call('objects/list').get('objects')
    if not isinstance(available, list):
        raise SafetyError('Printer object list is unavailable')
    wanted = ('webhooks', 'print_stats', 'toolhead', 'gcode_move', 'motion_report',
              'heater_bed', 'extruder', 'bed_mesh')
    objects = {name: None for name in wanted if name in available}
    if 'configfile' in available:
        objects['configfile'] = ['settings']
    for name in available:
        if isinstance(name, str) and name.lower() == 'gcode_macro xyz_ready':
            objects[name] = ['safe_z']
    return objects


def capabilities(api):
    """Read only the configuration needed for motion; never return full config."""
    available = api.call('objects/list').get('objects', [])
    if 'configfile' not in available:
        raise SafetyError('Firmware motion configuration is unavailable')
    result = api.call('objects/query', {'objects': {'configfile': ['settings']}})
    settings = result.get('status', {}).get('configfile', {}).get('settings')
    if not isinstance(settings, dict):
        raise SafetyError('Firmware motion configuration is unavailable')
    printer = settings.get('printer', {})
    report = {'ok': True, 'simulated': False, 'motion_limits': {
        name: printer.get(name) for name in ('kinematics', 'max_velocity', 'max_accel',
                                           'max_z_velocity', 'max_z_accel')},
        'homing_configuration': {name.lower(): value for name, value in settings.items() if name.lower() in
                                 ('homing_override', 'safe_z_home', 'gcode_macro g28')},
        'homing_macros': {name.lower(): value for name, value in settings.items() if name.lower() in
                         ('gcode_macro _home_x', 'gcode_macro _home_y', 'gcode_macro _home_z',
                          'gcode_macro _if_move_x', 'gcode_macro _if_move_y', 'gcode_macro _if_home_z',
                          'gcode_macro bed_mesh_save', 'gcode_macro xyz_ready')},
        'steppers': {axis: {field: settings.get('stepper_' + axis, {}).get(field)
                           for field in ('position_min', 'position_max', 'position_endstop', 'homing_speed')}
                     for axis in ('x', 'y', 'z')}}
    profile = {key: report[key] for key in ('homing_configuration', 'homing_macros', 'steppers')}
    report['homing_signature'] = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return report


def snapshot(api, objects, timeout=RESPONSE_TIMEOUT):
    result = api.call('objects/query', {'objects': objects}, timeout=timeout)
    if not isinstance(result.get('status'), dict):
        raise SafetyError('Printer status is unavailable')
    eventtime = number(result.get('eventtime'), 'Status timestamp')
    config = result['status'].pop('configfile', None)
    if config is not None:
        settings = config.get('settings', {}).get('printer', {})
        result['status']['motion_limits'] = {key: settings.get(key) for key in
            ('max_velocity', 'max_z_velocity', 'max_accel', 'max_z_accel')}
    for name in list(result['status']):
        if name.lower() == 'gcode_macro xyz_ready':
            result['status']['homing_lift_mm'] = result['status'].pop(name).get('safe_z')
    return result['status'], eventtime


def polling_objects(objects, homing=False):
    """Full configuration belongs in preflight, not the 100 ms telemetry loop."""
    return {name: fields for name, fields in objects.items() if name != 'configfile'
            and (homing or name.lower() != 'gcode_macro xyz_ready')}


def execution_snapshot(api, objects, motion_limits, timeout=RESPONSE_TIMEOUT):
    state, stamp = snapshot(api, objects, timeout)
    # Config-file limits stay fixed until firmware restart/reload. Dynamic
    # toolhead velocity/acceleration and all motion guards are still read live.
    if motion_limits is not None:
        state['motion_limits'] = dict(motion_limits)
    return state, stamp


def diagnostic_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [diagnostic_value(item) for item in value]
    return value


def telemetry_record(state, elapsed, query_seconds, stamp):
    return {'elapsed_seconds': elapsed, 'query_seconds': query_seconds, 'eventtime': stamp,
            'queued_position': diagnostic_value(state.get('toolhead', {}).get('position')),
            'live_position': diagnostic_value(state.get('motion_report', {}).get('live_position')),
            'live_velocity': diagnostic_value(state.get('motion_report', {}).get('live_velocity')),
            'homed_axes': state.get('toolhead', {}).get('homed_axes'),
            'firmware_state': state.get('webhooks', {}).get('state')}


def emergency_stop(path=SOCKET_PATH):
    api = None
    try:
        api = API(path)
        api.call('emergency_stop')
        return True
    except Exception:
        return False
    finally:
        if api is not None:
            api.close()


def execute(api, objects, request, stop, poll_interval=POLL_INTERVAL):
    """One queued script; monitoring uses the same local socket, without SSH gaps."""
    state, stamp = snapshot(api, objects)
    if 'expected_limits_mm' in request and request['expected_limits_mm'] != travel_limits(state):
        raise SafetyError('Firmware travel limits changed since preview; preview again')
    prepared = (plan_absolute(state, request['target'], request['speed_mm_s'])
                if request.get('action') == 'absolute-move' else
                plan_xy_start(state) if request.get('action') == 'xy-start' else
                plan(state, request['steps'], request['speed_mm_s'], request.get('z_speed_mm_s', DEFAULT_Z_SPEED)))
    if not same_position(prepared['start'], request['expected_start']):
        raise SafetyError('Position changed since preview; preview again')
    if not prepared['targets']:
        return {'ok': True, 'reported_position': prepared['start'], 'commands': 0,
                'simulated': False, 'message': 'Already at target; no motion sent'}
    motion_limits = state.get('motion_limits')
    telemetry_objects = polling_objects(objects)
    samples = []
    max_query_seconds = 0.0
    queried_at = None
    attempted = False
    try:
        dispatched_at = time.monotonic()
        deadline = dispatched_at + prepared['timeout']
        startup_deadline = dispatched_at + STARTUP_ZERO_WINDOW
        startup_open = True
        startup_zero_seen = False
        attempted = True
        move_id = api.send('gcode/script', {'script': prepared['script']})
        completed_at = None
        while True:
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('Sequence exceeded watchdog deadline')
            query_timeout = RESPONSE_TIMEOUT
            if startup_open:
                remaining = startup_deadline - now
                if remaining <= 0:
                    raise SafetyError('No confirmed live position within the 0.5-second startup window')
                query_timeout = min(query_timeout, remaining)
            queried_at = time.monotonic()
            state, next_stamp = execution_snapshot(api, telemetry_objects, motion_limits, timeout=query_timeout)
            observed_at = time.monotonic()
            query_seconds = observed_at - queried_at
            queried_at = None
            max_query_seconds = max(max_query_seconds, query_seconds)
            samples.append(telemetry_record(state, observed_at - dispatched_at, query_seconds, next_stamp))
            samples = samples[-8:]
            if next_stamp <= stamp:
                raise SafetyError('Status timestamp did not advance')
            stamp = next_stamp
            if startup_open and observed_at >= startup_deadline:
                raise SafetyError('No confirmed live position within the 0.5-second startup window')
            allow_zero = startup_open and startup_zero_report(state)
            if allow_zero:
                startup_zero_seen = True
            else:
                startup_open = False
            queued, live = check_execution_state(state, prepared, allow_startup_zero=allow_zero)
            if move_id in api.replies:
                api.result(move_id)  # M400 completed, or Klipper rejected the script.
                completed_at = time.monotonic()
            if completed_at is not None:
                if (not allow_zero and same_position(queued, prepared['end']) and same_position(live, prepared['end'])
                        and state['motion_report']['live_velocity'] <= 0.001):
                    check_execution_state(state, prepared, stationary=True)
                    return {'ok': True, 'reported_position': live, 'commands': len(prepared['targets']),
                            'watchdog': 'active during execution', 'simulated': False,
                            'startup_zero_report_seen': startup_zero_seen}
                if time.monotonic() - completed_at > 1.5:
                    raise SafetyError('Final reported position does not match the plan')
            delay = poll_interval
            if startup_open:
                delay = min(delay, max(0.0, startup_deadline - time.monotonic()))
            time.sleep(delay)
    except BaseException as exc:
        if not attempted:
            raise
        pending_query_seconds = time.monotonic() - queried_at if queried_at is not None else None
        confirmed = stop()
        return {'ok': False, 'error': str(exc) or type(exc).__name__,
                'command_may_have_executed': True, 'stop_requested': True,
                'stop_confirmed': confirmed,
                'startup_zero_report_seen': startup_zero_seen,
                'diagnostics': {'startup_window_seconds': STARTUP_ZERO_WINDOW,
                                'start': prepared['start'], 'target': prepared['end'],
                                'max_query_seconds': max_query_seconds,
                                'pending_query_seconds': pending_query_seconds,
                                'recent_telemetry': samples},
                'message': 'Inspect printer; manual recovery required. Do not automatically retry.'}


def homing_state(state, stationary=False):
    """Homing uses firmware travel bounds, including endstop parking coordinates.

    This separate check never relaxes the ordinary movement policy.
    """
    if state.get('webhooks', {}).get('state') != 'ready':
        raise SafetyError('Klipper is not ready')
    if state.get('print_stats', {}).get('state') not in ('standby', 'complete', 'cancelled'):
        raise SafetyError('Printer must be idle before homing')
    for name in ('heater_bed', 'extruder'):
        if number(state.get(name, {}).get('target'), name + ' target') != 0:
            raise SafetyError('Heater targets must be zero')
    th, move, live = state['toolhead'], state['gcode_move'], state['motion_report']
    if number(move.get('speed_factor'), 'Speed override') != 1:
        raise SafetyError('Speed override must be 100%')
    if any(xyz(move.get('homing_origin'), 'Homing origin')):
        raise SafetyError('Nonzero coordinate offsets are unsupported')
    homed = th.get('homed_axes')
    if not isinstance(homed, str) or len(set(homed)) != len(homed) or not set(homed) <= set('xyz'):
        raise SafetyError('Invalid homing report')
    queued = xyz(th.get('position'), 'Toolhead position')
    position = xyz(live.get('live_position'), 'Live position')
    minimum, maximum = xyz(th.get('axis_minimum'), 'Firmware minimum'), xyz(th.get('axis_maximum'), 'Firmware maximum')
    for point in (queued, position):
        if any(not minimum[i] - REPORT_ROUNDOFF <= point[i] <= maximum[i] + REPORT_ROUNDOFF for i in range(3)):
            raise SafetyError('Homing position is outside firmware travel limits')
    velocity = number(live.get('live_velocity'), 'Live velocity')
    limit, _ = firmware_speed_limits(state)
    if limit <= 0 or velocity < -REPORT_ROUNDOFF or velocity > limit + 0.01:
        raise SafetyError('Invalid velocity during homing')
    if stationary and (velocity > 0.001 or not same_position(queued, position)):
        raise SafetyError('Printer is moving or live status has not settled')
    return queued


def prepare_home(api, state, axes):
    if (not isinstance(axes, list) or not axes
            or any(not isinstance(axis, str) or axis not in ('x', 'y', 'z') for axis in axes)
            or len(set(axes)) != len(axes)):
        raise SafetyError('Homing axes must be a nonempty unique list of x, y, z')
    report = capabilities(api)
    if report['homing_signature'] != AUDITED_HOMING_SIGNATURE:
        raise SafetyError('Homing configuration changed or is unsupported; operator review required')
    start = homing_state(state, stationary=True)
    if number(state.get('homing_lift_mm'), 'Homing lift') != 5:
        raise SafetyError('Homing Z-lift variable changed; operator review required')
    return {'axes': sorted(axes), 'start': start, 'homed_axes': state['toolhead']['homed_axes'],
            'homing_signature': report['homing_signature'],
            'affected_axes': ['x', 'y', 'z'], 'firmware_routine': True,
            'message': 'Firmware homing may lift Z 5 mm before homing, move X/Y at 80–100 mm/s, '
                       'and save/clear/reload mesh. Review the whole physical homing path.'}


def execute_home(api, objects, request, stop, poll_interval=POLL_INTERVAL):
    state, stamp = snapshot(api, objects)
    prepared = prepare_home(api, state, request['axes'])
    if (prepared['homing_signature'] != request['homing_signature']
            or prepared['homed_axes'] != request['expected_homed_axes']
            or not same_position(prepared['start'], request['expected_start'])):
        raise SafetyError('Homing state changed since preview; preview again')
    start_e = number(state['gcode_move']['position'][3], 'Extruder position')
    motion_limits = state.get('motion_limits')
    telemetry_objects = polling_objects(objects, homing=True)
    attempted = False
    try:
        deadline = time.monotonic() + 180
        attempted = True
        ident = api.send('gcode/script', {'script': 'G28 ' + ' '.join(a.upper() for a in prepared['axes']) + '\nM400'})
        completed = False
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Homing exceeded watchdog deadline')
            state, next_stamp = execution_snapshot(api, telemetry_objects, motion_limits)
            if next_stamp <= stamp:
                raise SafetyError('Status timestamp did not advance')
            stamp = next_stamp
            homing_state(state)
            if number(state.get('homing_lift_mm'), 'Homing lift') != 5:
                raise SafetyError('Homing Z-lift variable changed during execution')
            if abs(number(state['gcode_move']['position'][3], 'Extruder position') - start_e) > 0.002:
                raise SafetyError('Unexpected extrusion during homing')
            if ident in api.replies:
                api.result(ident)
                completed = True
            if completed:
                position = homing_state(state, stationary=True)
                if not set(prepared['axes']) <= set(state['toolhead']['homed_axes']):
                    raise SafetyError('Firmware did not confirm requested axes homed')
                return {'ok': True, 'simulated': False, 'reported_position': position,
                        'homed_axes': state['toolhead']['homed_axes'],
                        'message': 'Homing completed; check controller bounds and mesh before movement'}
            time.sleep(poll_interval)
    except BaseException as exc:
        if not attempted:
            raise
        return {'ok': False, 'error': str(exc) or type(exc).__name__, 'command_may_have_executed': True,
                'stop_requested': True, 'stop_confirmed': stop(),
                'message': 'Inspect printer; manual recovery required. Do not automatically retry.'}


def validate_request(request):
    if not isinstance(request, dict):
        raise SafetyError('Invalid request')
    action = request.get('action')
    keys = {'status': {'action'}, 'capabilities': {'action'}, 'stop': {'action'}, 'mesh-clear': {'action'},
            'xy-start-preview': {'action'}, 'xy-start': {'action', 'expected_start'},
            'mesh-load': {'action', 'profile'},
            'absolute-preview': {'action', 'target', 'speed_mm_s'},
            'absolute-move': {'action', 'target', 'speed_mm_s', 'expected_start'},
            'home-preview': {'action', 'axes'},
            'home': {'action', 'axes', 'expected_start', 'expected_homed_axes', 'homing_signature'},
            'preview': {'action', 'steps', 'speed_mm_s'},
            'run': {'action', 'steps', 'speed_mm_s', 'expected_start'}}
    optional = {'z_speed_mm_s'} if action in ('preview', 'run') else set()
    if action in ('run', 'absolute-move', 'xy-start'):
        optional |= {'expected_limits_mm'}
    if action not in keys or not keys[action] <= set(request) or not set(request) <= keys[action] | optional:
        raise SafetyError('Unsupported action or fields')
    if action == 'mesh-load' and (not isinstance(request['profile'], str) or
                                 not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', request['profile'])):
        raise SafetyError('Invalid mesh profile name')
    return action


def handle(request, path=SOCKET_PATH, lock_path=LOCK_PATH):
    action = validate_request(request)
    if action == 'stop':
        confirmed = emergency_stop(path)
        return {'ok': confirmed, 'stop_requested': True, 'stop_confirmed': confirmed}
    lock = api = None
    try:
        if action in ('run', 'absolute-move', 'home', 'xy-start', 'mesh-clear', 'mesh-load'):
            # Protect against concurrent simplified runners; other printer apps
            # do not participate in this lock and must remain idle.
            lock = open(lock_path, 'a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SafetyError('Another simplified runner is already active')
        api = API(path)
        if action == 'capabilities':
            return capabilities(api)
        objects = query_objects(api)
        if action == 'home':
            return execute_home(api, objects, request, lambda: emergency_stop(path))
        if action in ('run', 'absolute-move', 'xy-start'):
            return execute(api, objects, request, lambda: emergency_stop(path))
        state, _ = snapshot(api, objects)
        if action == 'status':
            return {'ok': True, 'state': state, 'simulated': False}
        if action == 'home-preview':
            return {'ok': True, 'simulated': False, 'plan': prepare_home(api, state, request['axes'])}
        if action == 'xy-start-preview':
            return {'ok': True, 'plan': plan_xy_start(state), 'simulated': False}
        if action == 'absolute-preview':
            return {'ok': True, 'plan': plan_absolute(state, request['target'], request['speed_mm_s']),
                    'simulated': False}
        if action == 'preview':
            return {'ok': True, 'plan': plan(state, request['steps'], request['speed_mm_s'],
                                            request.get('z_speed_mm_s', DEFAULT_Z_SPEED)), 'simulated': False}
        # Mesh management is explicit; coordinate mismatch is expected before clear.
        check_state(state)
        live = state['motion_report']
        if live['live_velocity'] > 0.001 or not same_position(live['live_position'], state['toolhead']['position']):
            raise SafetyError('Mesh changes require a stationary printer')
        mesh_active(state.get('bed_mesh'))  # Require a valid report before changing it.
        script = 'BED_MESH_CLEAR' if action == 'mesh-clear' else 'BED_MESH_PROFILE LOAD=' + request['profile']
        try:
            api.call('gcode/script', {'script': script})
            after, _ = snapshot(api, objects)
        except Exception as exc:
            return {'ok': False, 'error': str(exc), 'command_may_have_executed': True}
        active = mesh_active(after.get('bed_mesh'))
        if action == 'mesh-clear' and active:
            raise SafetyError('Mesh still active after clear; inspect before retrying')
        if action == 'mesh-load' and (not active or after['bed_mesh']['profile_name'] != request['profile']):
            raise SafetyError('Unexpected loaded mesh; inspect before retrying')
        return {'ok': True, 'bed_mesh': after['bed_mesh'], 'simulated': False}
    finally:
        if api is not None:
            api.close()
        if lock is not None:
            lock.close()


def interrupted(signum, frame):
    raise InterruptedError('Watchdog received signal %s' % signum)


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        raw = sys.stdin.buffer.read(65537)
        if len(raw) > 65536:
            raise SafetyError('Request too large')
        result = handle(json.loads(raw))
    except Exception as exc:
        result = {'ok': False, 'error': str(exc)}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
