"""Version 1 JSON interface shared by MCP and the JSON-lines Terminal runner.

See docs/LLM_INTERFACE.md. Positions are absolute mm; speeds are mm/s.
No physical action occurs merely by importing this module.
"""
import argparse
import copy
import json
import os
import sys
import threading
import time
import uuid

from printer_client import MockPrinter, PrinterError, SSHPrinter, checked, run_prepared
from printer_policy import DEFAULT_SPEED, MAX_SPEED, MAX_Z_SPEED, SafetyError, absolute_target, number, travel_limits


class CommandError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def error_info(exc):
    if isinstance(exc, CommandError):
        return {'code': exc.code, 'message': str(exc), 'retryable': False, 'details': exc.details}
    message, details = str(exc), dict(getattr(exc, 'details', {}))
    try:
        backend = json.loads(message)
        if isinstance(backend, dict):
            message = str(backend.get('error', message))
            details = {key: backend[key] for key in
                       ('command_may_have_executed', 'stop_requested', 'stop_confirmed', 'diagnostics') if key in backend}
    except (ValueError, TypeError):
        pass
    code = 'PRINTER_ERROR'
    if details.get('command_may_have_executed') or 'outcome uncertain' in message.lower():
        code = 'OUTCOME_UNCERTAIN'
    elif 'changed since preview' in message:
        code = 'PREVIEW_CHANGED'
    elif 'outside' in message or 'travel limits' in message:
        code = 'OUT_OF_BOUNDS'
    elif 'Home all axes' in message or 'Home X and Y' in message:
        code = 'NOT_HOMED'
    elif 'already active' in message or 'moving' in message or 'must be idle' in message:
        code = 'BUSY'
    elif 'mesh' in message.lower():
        code = 'MESH_ACTIVE'
    elif 'Heater targets' in message:
        code = 'HEATING_ACTIVE'
    elif 'offset' in message.lower():
        code = 'COORDINATE_OFFSET'
    elif 'homing configuration' in message.lower() or 'homing z-lift' in message.lower():
        code = 'HOME_CONFIGURATION_REQUIRED'
    elif isinstance(exc, SafetyError):
        code = 'INVALID_ARGUMENT'
    elif isinstance(exc, (OSError, PrinterError)):
        code = 'CONNECTION_ERROR'
    return {'code': code, 'message': message, 'retryable': False, 'details': details}


class PrinterCommands:
    """One motion at a time; read/stop commands bypass the motion lock.

    Motion starts asynchronously and is polled with command=job. Real actions
    require the existing local PRINTER_ALLOW_MOTION=1 gate and explicit approval.
    Tokens are expiring and single-use; rejected busy commands are never queued.
    """
    def __init__(self, printer, motion_lock=None, on_invalidate=None, homing_mode=None):
        self._printer = printer
        self.motion_lock = motion_lock if motion_lock is not None else threading.Lock()
        self.lock = threading.RLock()
        self.tokens, self.jobs, self.workers = {}, {}, []
        self.active_job = None
        self.generation = 0
        self.on_invalidate = on_invalidate
        self.homing_mode = homing_mode or os.getenv('PRINTER_HOMING_MODE', 'manual')
        if self.homing_mode not in ('manual', 'firmware'):
            raise ValueError('PRINTER_HOMING_MODE must be manual or firmware')

    @property
    def printer(self):
        return self._printer() if callable(self._printer) else self._printer

    def response(self, request, status, message, data=None, error=None):
        return {'version': 1, 'id': request.get('id'), 'command': request.get('command'),
                'ok': error is None, 'simulated': self.printer.simulated, 'status': status,
                'message': message, 'data': data, 'error': error}

    def public_plan(self, kind, prepared):
        if kind == 'home':
            return {'axes': list(prepared['axes']), 'start_mm': dict(zip(('x', 'y', 'z'), prepared['start'])),
                    'affected_axes': list(prepared.get('affected_axes', ['x', 'y', 'z'])),
                    'coordinate_units': 'mm', 'speed_control': 'firmware', 'message': prepared['message']}
        return {'coordinate_mode': 'absolute', 'coordinate_units': 'mm', 'path': 'straight_line',
                'start_mm': dict(zip(('x', 'y', 'z'), prepared['start'])),
                'requested_target_mm': copy.deepcopy(prepared['target']),
                'target_mm': dict(zip(('x', 'y', 'z'), prepared['end'])),
                'speed_mm_s': prepared['speed_mm_s'], 'nominal_seconds': prepared['nominal_seconds'],
                'limits_mm': copy.deepcopy(prepared['firmware_limits_mm'])}

    def invalidate(self):
        with self.lock:
            self.tokens.clear()
            self.generation += 1
        if self.on_invalidate is not None:
            self.on_invalidate()

    def execute(self, request):
        context = request if isinstance(request, dict) else {}
        try:
            self.validate(request)
            result = self.dispatch(request)
            json.dumps(result, allow_nan=False)
            return result
        except (CommandError, SafetyError, PrinterError, ValueError, TypeError, KeyError, OSError) as exc:
            info = error_info(exc)
            if not isinstance(exc, (CommandError, SafetyError, PrinterError, OSError)):
                info = {'code': 'INVALID_ARGUMENT', 'message': str(exc), 'retryable': False, 'details': {}}
            # Correlation fields cannot echo arbitrary nested input.
            context = {key: value for key, value in context.items() if key in ('id', 'command')
                       and (value is None or isinstance(value, (str, int)) and not isinstance(value, bool))}
            return self.response(context, 'error', info['message'], error=info)
        except Exception:
            info = {'code': 'INTERNAL_ERROR', 'message': 'Internal controller error; no automatic retry',
                    'retryable': False, 'details': {}}
            return self.response({}, 'error', info['message'], error=info)

    def validate(self, request):
        if not isinstance(request, dict):
            raise CommandError('INVALID_COMMAND', 'Command must be a JSON object')
        allowed = {'move': {'target', 'speed_mm_s', 'preview', 'token', 'operator_confirmed'},
                   'home': {'axes', 'preview', 'token', 'operator_confirmed'},
                   'status': set(), 'position': set(), 'stop': set(), 'capabilities': set(),
                   'job': {'job_id'}}
        command = request.get('command')
        if not isinstance(command, str) or command not in allowed:
            raise CommandError('INVALID_COMMAND', 'Supported commands: ' + ', '.join(allowed))
        if set(request) - allowed[command] - {'command', 'id'}:
            raise CommandError('INVALID_COMMAND', 'Unsupported command fields')
        if 'id' in request and (isinstance(request['id'], bool) or not isinstance(request['id'], (str, int))):
            raise CommandError('INVALID_COMMAND', 'id must be a string or integer')
        for field in ('preview', 'operator_confirmed'):
            if field in request and not isinstance(request[field], bool):
                raise CommandError('INVALID_ARGUMENT', field + ' must be a JSON boolean')
        if 'token' in request and not isinstance(request['token'], str):
            raise CommandError('INVALID_ARGUMENT', 'token must be a string')
        if command == 'move':
            absolute_target(request.get('target'))
            if 'speed_mm_s' in request:
                value = number(request['speed_mm_s'], 'Speed')
                if not 0 < value <= min(MAX_SPEED, MAX_Z_SPEED):
                    raise CommandError('INVALID_ARGUMENT', 'Speed must be >0 and <=30 mm/s')
        if command == 'home':
            axes = request.get('axes', ['x', 'y', 'z'])
            if (not isinstance(axes, list) or not axes
                    or any(not isinstance(a, str) or a not in ('x', 'y', 'z') for a in axes)
                    or len(set(axes)) != len(axes)):
                raise CommandError('INVALID_ARGUMENT', 'axes must be a nonempty unique list of x, y, z')
        if command == 'job' and not isinstance(request.get('job_id'), str):
            raise CommandError('INVALID_ARGUMENT', 'job_id must be a string')
        if command in ('move', 'home') and request.get('preview', True) and 'token' in request:
            raise CommandError('INVALID_COMMAND', 'Use preview=false when executing a token')

    def dispatch(self, request):
        command, printer = request['command'], self.printer
        if command in ('status', 'position'):
            reply = checked(printer, {'action': 'status'})
            state = reply['state']
            position = dict(zip(('x', 'y', 'z'), state['motion_report']['live_position'][:3]))
            with self.lock:
                active = self.active_job
            data = {'position_mm': position, 'homed_axes': state['toolhead']['homed_axes'],
                    'travel_limits_mm': travel_limits(state), 'travel_limits_source': 'firmware',
                    'busy': self.motion_lock.locked() or state['motion_report']['live_velocity'] > 0.001,
                    'active_job_id': active, 'firmware_state': state['webhooks']['state']}
            if command == 'status':
                data['state'] = state
            return self.response(request, 'running' if data['busy'] else 'idle', 'Printer state read', data)
        if command == 'capabilities':
            reply = checked(printer, {'action': 'capabilities'})
            limits = travel_limits(checked(printer, {'action': 'status'})['state'])
            data = {'coordinate_mode': 'absolute', 'coordinate_units': 'mm', 'speed_units': 'mm/s',
                    'acceleration_units': 'mm/s^2', 'acceleration_control': 'firmware',
                    'travel_limits_mm': limits, 'travel_limits_source': 'firmware',
                    'software_limits': {'max_speed_mm_s': 30, 'default_speed_mm_s': DEFAULT_SPEED},
                    'firmware_limits': reply['motion_limits'], 'homing_mode': self.homing_mode,
                    'motion_enabled': printer.simulated or os.getenv('PRINTER_ALLOW_MOTION') == '1'}
            return self.response(request, 'completed', 'Motion capabilities read', data)
        if command == 'job':
            with self.lock:
                job = copy.deepcopy(self.jobs.get(request['job_id']))
            if job is None:
                raise CommandError('JOB_NOT_FOUND', 'Unknown or evicted job_id')
            return self.response(request, job['status'], job['message'], job, job['error'])
        if command == 'stop':
            self.invalidate()
            reply = printer.request({'action': 'stop'})
            if not reply.get('ok') or not reply.get('stop_confirmed'):
                raise CommandError('STOP_UNCONFIRMED', 'Emergency stop was not confirmed; inspect printer', reply)
            with self.lock:
                if self.active_job:
                    self.jobs[self.active_job]['stop_requested'] = True
            return self.response(request, 'stopped', 'Emergency shutdown confirmed; manual recovery required', reply)
        if command == 'home' and not printer.simulated and self.homing_mode != 'firmware':
            raise CommandError('HOME_CONFIGURATION_REQUIRED',
                               'Home using the touchscreen, then check status. Firmware homing requires '
                               'a local PRINTER_HOMING_MODE=firmware setting after reviewing its routine.')
        if not self.motion_lock.acquire(blocking=False):
            raise CommandError('BUSY', 'Another motion or homing operation is active; command was not queued')
        retained = False
        try:
            if request.get('preview', True):
                with self.lock:
                    generation = self.generation
                if command == 'move':
                    prepared = checked(printer, {'action': 'absolute-preview', 'target': request['target'],
                                       'speed_mm_s': request.get('speed_mm_s', DEFAULT_SPEED)})['plan']
                else:
                    prepared = checked(printer, {'action': 'home-preview',
                                       'axes': request.get('axes', ['x', 'y', 'z'])})['plan']
                token = uuid.uuid4().hex
                with self.lock:
                    if generation != self.generation:
                        raise CommandError('TOKEN_INVALID', 'Preview was invalidated by stop or homing; preview again')
                    now = time.monotonic()
                    self.tokens = {k: v for k, v in self.tokens.items() if v['expires'] > now}
                    if len(self.tokens) >= 100:
                        raise CommandError('TOO_MANY_PREVIEWS', 'Too many pending previews')
                    self.tokens[token] = {'kind': command, 'plan': prepared, 'expires': now + 120,
                                          'requested_speed': request.get('speed_mm_s', DEFAULT_SPEED)}
                return self.response(request, 'previewed', 'Review the path before confirming execution',
                                     {'token': token, 'expires_in_seconds': 120,
                                      'plan': self.public_plan(command, prepared)})
            if request.get('operator_confirmed') is not True:
                raise CommandError('CONFIRMATION_REQUIRED', 'Operator approval of the reviewed path is required')
            if not printer.simulated and os.getenv('PRINTER_ALLOW_MOTION') != '1':
                raise CommandError('MOTION_DISABLED', 'Real motion requires local PRINTER_ALLOW_MOTION=1')
            with self.lock:
                token = self.tokens.get(request.get('token'))
                if token is None or token['expires'] <= time.monotonic() or token['kind'] != command:
                    raise CommandError('TOKEN_INVALID', 'Preview token is missing, expired or already used')
                prepared = token['plan']
                if command == 'move':
                    if request['target'] != prepared['target'] or request.get('speed_mm_s', token['requested_speed']) != token['requested_speed']:
                        raise CommandError('PREVIEW_CHANGED', 'Target or speed differs from preview; preview again')
                elif sorted(request.get('axes', ['x', 'y', 'z'])) != prepared['axes']:
                    raise CommandError('PREVIEW_CHANGED', 'Homing axes differ from preview; preview again')
                self.tokens.pop(request['token'])
                job_id = uuid.uuid4().hex
                self.jobs = {k: v for k, v in self.jobs.items() if v['status'] == 'running'} if len(self.jobs) >= 100 else self.jobs
                self.jobs[job_id] = {'job_id': job_id, 'kind': command, 'status': 'running',
                                     'message': 'Operation running', 'result': None, 'error': None}
                self.active_job = job_id
            if command == 'home':
                self.invalidate()
            worker = threading.Thread(target=self._run_job, args=(printer, job_id, command, prepared), daemon=False)
            worker.start()
            self.workers = [item for item in self.workers if item.is_alive()]
            self.workers.append(worker)
            retained = True
            return self.response(request, 'accepted', 'Operation accepted; poll job_id for completion', {'job_id': job_id})
        finally:
            if not retained:
                self.motion_lock.release()

    def _run_job(self, printer, job_id, kind, prepared):
        try:
            if kind == 'move':
                result = run_prepared(printer, [], prepared['speed_mm_s'], prepared)
            else:
                request = {'action': 'home', 'axes': prepared['axes'], 'expected_start': prepared['start'],
                           'expected_homed_axes': prepared['homed_axes'], 'homing_signature': prepared['homing_signature']}
                try:
                    result = printer.request(request)
                except (PrinterError, OSError) as exc:
                    # Never retry uncertain homing. Try the independent stop path.
                    try:
                        stop = checked(printer, {'action': 'stop'})
                    except Exception:
                        stop = {'stop_confirmed': False}
                    raise CommandError('OUTCOME_UNCERTAIN', str(exc),
                                       {'command_may_have_executed': True, 'stop_requested': True,
                                        'stop_confirmed': stop.get('stop_confirmed', False)}) from exc
                if not result.get('ok'):
                    raise PrinterError(json.dumps(result))
            json.dumps(result, allow_nan=False)
            public = {'position_mm': dict(zip(('x', 'y', 'z'), result['reported_position']))}
            if 'homed_axes' in result:
                public['homed_axes'] = result['homed_axes']
            with self.lock:
                job = self.jobs[job_id]
                job.update(status='stopped' if job.get('stop_requested') else 'completed',
                           message='Operation stopped' if job.get('stop_requested') else 'Operation completed', result=public)
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                info = error_info(exc)
                job.update(status='stopped' if job.get('stop_requested') else 'error',
                           message=info['message'], error=info)
        finally:
            with self.lock:
                self.active_job = None
            self.motion_lock.release()

    def close(self):
        for worker in self.workers:
            worker.join()
        self.workers.clear()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def parse_command(raw):
    def invalid_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(raw, parse_constant=invalid_constant, object_pairs_hook=unique_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description='One JSON command per input line; one JSON response per output line.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--mock', action='store_true')
    modes.add_argument('--real', action='store_true')
    args = parser.parse_args(argv)
    mode = 'real' if args.real else 'mock' if args.mock else os.getenv('PRINTER_MODE', 'mock')
    try:
        if mode not in ('mock', 'real'):
            raise ValueError('PRINTER_MODE must be mock or real')
        controller = PrinterCommands(MockPrinter() if mode == 'mock' else SSHPrinter())
    except (ValueError, PrinterError) as exc:
        print(json.dumps({'version': 1, 'id': None, 'command': None, 'ok': False,
                          'simulated': mode != 'real', 'status': 'error', 'message': str(exc),
                          'data': None, 'error': error_info(exc)}, allow_nan=False), flush=True)
        return 1
    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            try:
                if len(raw.encode()) > 65536:
                    raise ValueError('Command exceeds 65536 bytes')
                result = controller.execute(parse_command(raw))
            except ValueError as exc:
                info = {'code': 'INVALID_JSON', 'message': str(exc), 'retryable': False, 'details': {}}
                result = controller.response({}, 'error', str(exc), error=info)
            print(json.dumps(result, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        if controller.motion_lock.locked():
            print(json.dumps(controller.execute({'command': 'stop'}), allow_nan=False), flush=True)
        return 130
    finally:
        # EOF waits for accepted work. Use stop explicitly before exiting to cancel.
        controller.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
