import json
import pytest
from main.batch.common import ROOT
from main.batch.__main__ import parser,create,round_tasks
from main.batch.user_config import resolve,prepare


def test_frozen_matrix(tmp_path):
    src=tmp_path/'config.json'; raw=json.loads((ROOT/'configs/closed_loop_batch.json').read_text());full=resolve(raw)
    assert len(full['seas'])*len(full['weather'])*len(full['methods'])*10==320
    raw.update(sea_levels=[6],weather=['night','fog'],methods=['C1']);raw['conditions']['6'].update(wind_mps=14.5,prediction_window_s=[3.1,5.9]);src.write_text(json.dumps(raw))
    out=prepare(src,tmp_path);src.write_text('{}');m=create(parser().parse_args(['plan','--output','outputs/batch/test','--batch-json',str(out/'batch_config.json')]))
    tasks=round_tasks(out,m);assert len(tasks)==20
    assert {(t['sea'],t['method'],t['group']['weather']) for t in tasks}=={(6,'C1','fog'),(6,'C1','night')}
    assert m['policy']['seas'][6]==raw['conditions']['6']
    raw['conditions']['6']['prediction_window_s']=[5.01,6]
    with pytest.raises(ValueError):resolve(raw)
