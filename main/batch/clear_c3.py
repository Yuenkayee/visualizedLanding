"""Submit the fixed clear-weather condition through the formal batch coordinator."""
import os
import sys

from .__main__ import main, parser


def main_clear():
    import matlab.engine
    names = sorted(matlab.engine.find_matlab())
    usual = [f'LandingBatch{i}' for i in range(1, 7)]
    sessions = os.getenv('LANDING_C3_SESSIONS') or ','.join(usual if set(usual) <= set(names) else names)
    a = parser().parse_args(['run', '--clear-c3-w10-h2', '--workers', '6', '--output', sys.argv[1],
                             '--matlab-sessions', sessions])
    print('MATLAB sessions:', sessions, flush=True)
    main(a, log_directory=True)


if __name__ == '__main__':
    main_clear()
