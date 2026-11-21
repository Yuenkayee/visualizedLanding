"""Check authorization in the six shared engines before dispatching any cases."""
import os
import sys


def check(api, requested):
    available = sorted(api.find_matlab())
    usual = [f'LandingBatch{i}' for i in range(1, 7)]
    names = ([s.strip() for s in requested.split(',')] if requested else
             usual if set(usual) <= set(available) else available)
    if len(names) != 6 or len(set(names)) != 6 or not set(names) <= set(available):
        raise ValueError(f'需要指定 6 个不同且可见的共享会话。可见：{available}；所选：{names}。可用 LANDING_C3_SESSIONS 指定。')
    failed = []
    for worker, name in enumerate(names, 1):
        try:
            eng = api.connect_matlab(name)
            try:
                for feature in ('Simulink', 'Control_Toolbox', 'Robust_Toolbox'):
                    ok, message = eng.license('checkout', feature, nargout=2)
                    if not ok:
                        raise RuntimeError(f'{feature}: {message}')
                eng.eval('assert(isa(ss(-1,1,1,0),\'ss\'));', nargout=0)
                eng.load_system('simulink', nargout=0)
                if not eng.which('hinfsyn'):
                    raise RuntimeError('未找到 hinfsyn')
            finally:
                eng.quit()  # Disconnect the client of this already shared engine.
        except Exception as exc:
            failed.append(name)
            print(f'[授权失败] worker {worker} | {name}: {exc}', file=sys.stderr, flush=True)
        else:
            print(f'[授权通过] worker {worker} | {name}', file=sys.stderr, flush=True)
    if failed:
        raise RuntimeError('未启动仿真；请修复会话授权后重试：' + ', '.join(failed))
    return names


if __name__ == '__main__':
    import matlab.engine
    print(','.join(check(matlab.engine, os.getenv('LANDING_C3_SESSIONS'))))
