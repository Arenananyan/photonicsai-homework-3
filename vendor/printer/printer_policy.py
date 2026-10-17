"""Shared, pure movement checks. Python 3.8 compatible for the printer."""
import math

MAX_SPEED = 30.0
MAX_Z_SPEED = 30.0
DEFAULT_SPEED = 20.0
DEFAULT_Z_SPEED = 20.0
POSITION_TOLERANCE = 0.002
# Arithmetic noise seen in the firmware's interpolated idle reports. Applies
# only to telemetry; planned/rounded command targets retain strict bounds.
REPORT_ROUNDOFF = 1e-9


class SafetyError(ValueError):
    pass


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyError(name + ' must be a finite number')
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise SafetyError(name + ' must be a finite number')
    if not math.isfinite(value):
        raise SafetyError(name + ' must be a finite number')
    return value


def xyz(value, name):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise SafetyError(name + ' is missing or invalid')
    return [number(v, name) for v in value[:3]]


def obj(state, name):
    if not isinstance(state, dict) or not isinstance(state.get(name), dict):
        raise SafetyError(name + ' status is unavailable')
    return state[name]


def same_position(actual, expected):
    return all(abs(a - b) <= POSITION_TOLERANCE
               for a, b in zip(xyz(actual, 'Position'), xyz(expected, 'Expected position')))


def travel_limits(state):
    """Authoritative limits in configured machine coordinates; no guessed fallback."""
    th = obj(state, 'toolhead')
    minimum = xyz(th.get('axis_minimum'), 'Firmware minimum')
    maximum = xyz(th.get('axis_maximum'), 'Firmware maximum')
    if any(not minimum[i] < maximum[i] for i in range(3)):
        raise SafetyError('Firmware travel limits are missing or invalid')
    return {axis: [minimum[i], maximum[i]] for i, axis in enumerate(('x', 'y', 'z'))}


def check_xy(position, bounds, tolerance=0.0):
    position = xyz(position, 'Position')
    for value, low, high, name in ((position[0], bounds[0], bounds[1], 'X'),
                                    (position[1], bounds[2], bounds[3], 'Y')):
        if not low - tolerance <= value <= high + tolerance:
            raise SafetyError('%s=%.6f is outside allowed travel bounds %g..%g mm' % (name, value, low, high))


def check_z(position, bounds, tolerance=0.0):
    value = xyz(position, 'Position')[2]
    if not bounds[0] - tolerance <= value <= bounds[1] + tolerance:
        raise SafetyError('Z=%.6f is outside firmware travel bounds %g..%g mm' % (value, bounds[0], bounds[1]))


def check_target(position, limits):
    check_xy(position, limits['x'] + limits['y'])
    check_z(position, limits['z'])


def mesh_active(mesh):
    if not isinstance(mesh, dict) or not isinstance(mesh.get('profile_name'), str):
        raise SafetyError('Bed-mesh status is unavailable')
    matrix = mesh.get('mesh_matrix')
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise SafetyError('Bed-mesh matrix is invalid')
    if matrix and any(len(row) != len(matrix[0]) for row in matrix):
        raise SafetyError('Bed-mesh matrix is ragged')
    for row in matrix:
        for v in row:
            number(v, 'Bed-mesh point')
    return bool(mesh['profile_name']) or any(len(row) for row in matrix)


def startup_zero_report(state):
    report = obj(state, 'motion_report')
    return (xyz(report.get('live_position'), 'Live position') == [0.0, 0.0, 0.0]
            and number(report.get('live_velocity'), 'Live velocity') == 0.0)


def check_state(state, start_z=None, stationary=False, z_range=None, speed_limit=MAX_SPEED,
                allow_startup_zero=False, required_axes='xyz', bounds=None, z_bounds=None):
    """Checks both interpolated live position and queued toolhead position."""
    if obj(state, 'webhooks').get('state') != 'ready':
        raise SafetyError('Klipper is not ready')
    if obj(state, 'print_stats').get('state') not in ('standby', 'complete', 'cancelled'):
        raise SafetyError('Printer must be idle, not printing or paused')
    th = obj(state, 'toolhead')
    limits = travel_limits(state)
    bounds = limits['x'] + limits['y'] if bounds is None else bounds
    z_bounds = limits['z'] if z_bounds is None else z_bounds
    homed = th.get('homed_axes')
    if (not isinstance(homed, str) or len(set(homed)) != len(homed)
            or not set(homed) <= set('xyz') or not set(required_axes) <= set(homed)):
        raise SafetyError('Home %s using the printer controls before running; placing an axis by hand does not home it' %
                          ('X and Y' if required_axes == 'xy' else 'all axes'))
    if 'z' not in homed and mesh_active(state.get('bed_mesh')):
        raise SafetyError('X/Y-only movement with Z unhomed requires an inactive bed mesh')
    for name in ('heater_bed', 'extruder'):
        if name == 'heater_bed' or name in state:
            if number(obj(state, name).get('target'), name + ' target') != 0:
                raise SafetyError('Heater targets must be zero')
    move = obj(state, 'gcode_move')
    if any(xyz(move.get('homing_origin'), 'Homing origin')):
        raise SafetyError('Nonzero coordinate offsets are unsupported')
    if number(move.get('speed_factor'), 'Speed override') != 1.0:
        raise SafetyError('Speed override must be 100%')
    machine = xyz(move.get('position'), 'G-code machine position')
    gcode = xyz(move.get('gcode_position'), 'G-code position')
    if any(abs(a - b) > 1e-6 for a, b in zip(machine, gcode)):
        raise SafetyError('G92 coordinate offsets are unsupported')
    queued = xyz(th.get('position'), 'Toolhead position')
    live_obj = obj(state, 'motion_report')
    live = xyz(live_obj.get('live_position'), 'Live position')
    velocity = number(live_obj.get('live_velocity'), 'Live velocity')
    if -REPORT_ROUNDOFF <= velocity < 0:
        velocity = 0.0
    if velocity < 0 or velocity > speed_limit + 0.01:
        raise SafetyError('Live velocity exceeds the allowed %g mm/s' % speed_limit)
    mins, maxs = xyz(th.get('axis_minimum'), 'Firmware minimum'), xyz(th.get('axis_maximum'), 'Firmware maximum')
    for label, point in (('queued', queued), ('live', live)):
        check_xy(point, bounds, REPORT_ROUNDOFF)
        check_z(point, z_bounds, REPORT_ROUNDOFF)
        for i in range(3):
            if not mins[i] < maxs[i] or not mins[i] - REPORT_ROUNDOFF <= point[i] <= maxs[i] + REPORT_ROUNDOFF:
                raise SafetyError('Reported position is outside firmware travel limits')
        missing_live = (label == 'live' and allow_startup_zero and not stationary
                        and point == [0.0, 0.0, 0.0] and velocity == 0.0)
        if not missing_live and start_z is not None and abs(point[2] - start_z) > POSITION_TOLERANCE:
            raise SafetyError('Unexpected Z change: %s Z=%.6f mm; expected Z=%.6f mm; '
                              'queued XYZ=%r; live XYZ=%r; live velocity=%r mm/s' %
                              (label, point[2], start_z, queued, live, velocity))
        if not missing_live and z_range is not None and not z_range[0] - POSITION_TOLERANCE <= point[2] <= z_range[1] + POSITION_TOLERANCE:
            raise SafetyError('Reported Z is outside the planned Z range')
    if stationary:
        if velocity > 0.001 or not same_position(live, queued):
            raise SafetyError('Printer is moving, has queued motion, or live status has not settled')
        if any(abs(a - b) > 1e-6 for a, b in zip(queued, machine)):
            raise SafetyError('Toolhead and G-code positions disagree; check bed mesh with --status')
    return queued, live


def check_execution_state(state, prepared, stationary=False, allow_startup_zero=False):
    if 'firmware_limits_mm' in prepared and travel_limits(state) != prepared['firmware_limits_mm']:
        raise SafetyError('Firmware travel limits changed during execution')
    xy_limit, z_limit = firmware_speed_limits(state)
    if prepared['speed_mm_s'] > xy_limit or prepared['moves_z'] and prepared['z_speed_mm_s'] > z_limit:
        raise SafetyError('Speed exceeds verified firmware velocity limits during execution')
    if prepared.get('absolute') and mesh_active(state.get('bed_mesh')):
        raise SafetyError('Absolute movement requires an inactive bed mesh')
    if prepared.get('xy_start') and mesh_active(state.get('bed_mesh')):
        raise SafetyError('XY entry requires an inactive bed mesh')
    return check_state(state, start_z=prepared['start'][2] if not prepared['moves_z'] else None,
                       stationary=stationary, z_range=prepared['z_range'],
                       speed_limit=max(prepared['step_speeds_mm_s'], default=prepared['speed_mm_s']),
                       allow_startup_zero=allow_startup_zero,
                       required_axes='xyz' if prepared['moves_z'] or prepared.get('absolute') else 'xy',
                       bounds=prepared['bounds'], z_bounds=prepared['z_bounds'])


def firmware_speed_limits(state):
    """Require verified per-axis firmware caps before using speeds above 20."""
    settings = state.get('motion_limits')
    if settings is None:
        return 20.0, 20.0
    if not isinstance(settings, dict):
        raise SafetyError('Firmware motion limits are unavailable')
    values = {key: number(settings.get(key), 'Firmware ' + key) for key in
              ('max_velocity', 'max_z_velocity', 'max_accel', 'max_z_accel')}
    if any(value <= 0 for value in values.values()):
        raise SafetyError('Firmware motion limits must be positive')
    dynamic = state.get('toolhead', {}).get('max_velocity', values['max_velocity'])
    acceleration = number(state.get('toolhead', {}).get('max_accel', values['max_accel']), 'Firmware current acceleration')
    if not 0 < acceleration <= values['max_accel']:
        raise SafetyError('Current acceleration exceeds verified firmware acceleration limit')
    xy = min(values['max_velocity'], number(dynamic, 'Firmware maximum velocity'))
    return xy, min(xy, values['max_z_velocity'])


def absolute_target(target, limits=None):
    """Validate millimetres in the configured machine origin, before rounding."""
    if not isinstance(target, dict) or set(target) != {'x', 'y', 'z'}:
        raise SafetyError('Target must contain exactly x, y and z in millimetres')
    requested = [number(target[axis], 'Target ' + axis.upper()) for axis in ('x', 'y', 'z')]
    rounded = [float('%.3f' % value) for value in requested]
    if limits is not None:
        for point in (requested, rounded):
            check_target(point, limits)
    return requested, rounded


def plan_absolute(state, target, speed):
    """One straight XYZ move; all axes homed, no mesh or coordinate resets."""
    limits = travel_limits(state)
    requested, end = absolute_target(target, limits)
    speed = number(speed, 'Speed')
    limit = min(MAX_SPEED, MAX_Z_SPEED)
    if not 0 < speed <= limit:
        raise SafetyError('Absolute speed must be >0 and <=%g mm/s' % limit)
    feed = float('%.6f' % (speed * 60))
    if feed <= 0 or feed > limit * 60:
        raise SafetyError('Rounded feed rate is invalid')
    speed = feed / 60
    start, _ = check_state(state, stationary=True, required_axes='xyz')
    xy_limit, z_limit = firmware_speed_limits(state)
    supported = z_limit if start[2] != end[2] else xy_limit
    if speed > supported:
        raise SafetyError('Absolute speed exceeds verified firmware velocity limit %g mm/s' % supported)
    if 'max_velocity' in state['toolhead']:
        firmware_speed = number(state['toolhead']['max_velocity'], 'Firmware maximum velocity')
        if firmware_speed <= 0 or speed > firmware_speed:
            raise SafetyError('Absolute speed exceeds firmware maximum velocity')
    if mesh_active(state.get('bed_mesh')):
        raise SafetyError('Absolute movement requires an inactive bed mesh; '
                          'run ./printerctl mesh-clear and confirm CLEAR, then preview again')
    minimum = xyz(state['toolhead']['axis_minimum'], 'Firmware minimum')
    maximum = xyz(state['toolhead']['axis_maximum'], 'Firmware maximum')
    for point in (requested, end):
        if any(not minimum[i] <= point[i] <= maximum[i] for i in range(3)):
            raise SafetyError('Target exceeds firmware travel limits')
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))
    nominal = distance / speed
    timeout = 20 + nominal * 2 + (0.1 if distance else 0)
    if timeout > 300:
        raise SafetyError('Movement is too long/slow for the 300-second watchdog deadline')
    targets = [end] if distance else []
    return {'absolute': True, 'target': dict(zip(('x', 'y', 'z'), requested)),
            'start': start, 'end': end, 'targets': targets, 'bounds': limits['x'] + limits['y'],
            'z_bounds': limits['z'], 'firmware_limits_mm': limits,
            'z_range': [min(start[2], end[2]), max(start[2], end[2])],
            'moves_z': start[2] != end[2], 'speed_mm_s': speed, 'z_speed_mm_s': speed,
            'step_speeds_mm_s': [speed] if distance else [], 'nominal_seconds': nominal,
            'timeout': timeout,
            'script': ('G90\nG1 X%.3f Y%.3f Z%.3f F%.6f\nM400' % (*end, feed)) if distance else ''}


def plan(state, steps, speed, z_speed=DEFAULT_Z_SPEED):
    speed = number(speed, 'Speed')
    if not 0 < speed <= MAX_SPEED:
        raise SafetyError('Speed must be >0 and <=%g mm/s' % MAX_SPEED)
    feed = float('%.6f' % (speed * 60))
    if feed <= 0 or feed > MAX_SPEED * 60:
        raise SafetyError('Rounded feed rate is invalid')
    speed = feed / 60
    z_speed = number(z_speed, 'Z speed')
    if not 0 < z_speed <= MAX_Z_SPEED:
        raise SafetyError('Z speed must be >0 and <=%g mm/s' % MAX_Z_SPEED)
    z_feed = float('%.6f' % (z_speed * 60))
    if z_feed <= 0 or z_feed > MAX_Z_SPEED * 60:
        raise SafetyError('Rounded Z feed rate is invalid')
    z_speed = z_feed / 60
    xy_limit, z_limit = firmware_speed_limits(state)
    if speed > xy_limit or z_speed > z_limit:
        raise SafetyError('Speed exceeds verified firmware velocity limits')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 1000:
        raise SafetyError('Provide 1..1000 movement steps')
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'axis', 'distance_mm'} or step['axis'] not in ('x', 'y', 'z', 'xy'):
            raise SafetyError('Each step needs axis x/y/z with numeric distance_mm, or xy with [dx,dy]')
    moves_z = any(step['axis'] == 'z' for step in steps)
    limits = travel_limits(state)
    start, _ = check_state(state, stationary=True, required_axes='xyz' if moves_z else 'xy')
    current = list(start)
    minimum = xyz(state['toolhead']['axis_minimum'], 'Firmware minimum')
    maximum = xyz(state['toolhead']['axis_maximum'], 'Firmware maximum')
    targets, lines, nominal, step_speeds = [], ['G90'], 0.0, []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'axis', 'distance_mm'} or step['axis'] not in ('x', 'y', 'z', 'xy'):
            raise SafetyError('Each step needs axis x/y/z with numeric distance_mm, or xy with [dx,dy]')
        target = list(current)
        if step['axis'] == 'xy':
            delta = step['distance_mm']
            if not isinstance(delta, list) or len(delta) != 2:
                raise SafetyError('XY distance_mm must be [dx,dy]')
            dx, dy = [number(value, 'XY distance') for value in delta]
            if dx == 0 and dy == 0:
                raise SafetyError('Step distance must be nonzero')
            target[0] += dx
            target[1] += dy
            i = 0
        else:
            delta = number(step['distance_mm'], 'Distance')
            if delta == 0:
                raise SafetyError('Step distance must be nonzero')
            i = ('x', 'y', 'z').index(step['axis'])
            target[i] += delta
        rounded = list(target)
        indices = (2,) if i == 2 else (0, 1)
        for j in indices:
            rounded[j] = float('%.3f' % target[j])
        for point in (target, rounded):
            check_target(point, limits)
            if any(not minimum[j] <= point[j] <= maximum[j] for j in range(3)):
                raise SafetyError('Target exceeds firmware travel limits')
        target = rounded
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, current)))
        if distance == 0:
            raise SafetyError('Step rounds to zero at 0.001 mm resolution')
        step_speed = z_speed if i == 2 else speed
        nominal += distance / step_speed
        step_speeds.append(step_speed)
        targets.append(target)
        if i == 2:
            lines.append('G1 Z%.3f F%.6f' % (target[2], z_feed))
        else:
            lines.append('G1 X%.3f Y%.3f F%.6f' % (target[0], target[1], feed))
        current = target
    timeout = 20 + nominal * 2 + len(steps) * 0.1
    if timeout > 300:
        raise SafetyError('Sequence is too long/slow for the 300-second watchdog deadline')
    lines.append('M400')
    heights = [start[2]] + [target[2] for target in targets]
    return {'start': start, 'targets': targets, 'end': current, 'bounds': limits['x'] + limits['y'],
            'z_bounds': limits['z'], 'firmware_limits_mm': limits,
            'z_range': [min(heights), max(heights)],
            'moves_z': moves_z,
            'z_speed_mm_s': z_speed, 'step_speeds_mm_s': step_speeds,
            'speed_mm_s': speed, 'nominal_seconds': nominal, 'timeout': timeout,
            'script': '\n'.join(lines)}


def plan_xy_start(state):
    """Separate operator-confirmed entry from the observed XY homing area.

    No caller-supplied bounds, speed, destination, homing or coordinate reset.
    Ordinary planning uses firmware limits, including valid negative coordinates.
    This optional convenience command retains its original narrow entry corridor.
    """
    start, _ = check_state(state, stationary=True, required_axes='xy',
                           bounds=(-12.0, 0.0, -15.0, 0.0), speed_limit=5.0)
    if mesh_active(state.get('bed_mesh')):
        raise SafetyError('XY entry requires an inactive bed mesh')
    if start[:2] == [0.0, 0.0]:
        raise SafetyError('Already at X=0 Y=0; use ./printerctl --preview for your steps')
    end = [0.0, 0.0, start[2]]
    limits = travel_limits(state)
    minimum = xyz(state['toolhead']['axis_minimum'], 'Firmware minimum')
    maximum = xyz(state['toolhead']['axis_maximum'], 'Firmware maximum')
    if any(not minimum[i] <= end[i] <= maximum[i] for i in range(3)):
        raise SafetyError('Entry target exceeds firmware travel limits')
    nominal = math.hypot(start[0], start[1]) / 5.0
    return {'start': start, 'targets': [end], 'end': end,
            'bounds': [start[0], 0.0, start[1], 0.0],
            'z_bounds': limits['z'], 'firmware_limits_mm': limits, 'z_range': [start[2], start[2]],
            'moves_z': False, 'xy_start': True, 'z_speed_mm_s': MAX_Z_SPEED,
            'step_speeds_mm_s': [5.0], 'speed_mm_s': 5.0,
            'nominal_seconds': nominal, 'timeout': 20.0 + nominal * 2,
            'script': 'G90\nG1 X0.000 Y0.000 F300.000000\nM400'}
