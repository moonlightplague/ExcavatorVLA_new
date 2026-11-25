"""Check real-cache head-only optimization, frozen outputs and prefix setup."""
import argparse,copy,importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace


def check(output):
    spec=importlib.util.spec_from_file_location('head',Path(__file__).with_name('train_act_conservative_head.py'))
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    original=run.common.metadata()
    def metadata():
        value=copy.deepcopy(original);value['split_manifest']=dict(train=[0],validation=[1],test=[]);return value
    run.common.metadata=metadata
    with tempfile.TemporaryDirectory(prefix='act_conservative_head_check_') as temp:
        run.train(SimpleNamespace(output=Path(temp),steps=2))
        result=run.read(Path(temp)/'training_result.json')
        assert result['strict_reload']=='passed' and result['frozen_model_unchanged']
        assert result['validation']['source_deviation']['swing_max_normalized_difference']==0
        text=run.evaluation_script(Path(temp),result['checkpoint'],[50,112],2000)
        compile(text,'prefix_evaluation','exec')
        assert "'--steps','2000'" in text and 'ids=[50, 112]' in text and 'episodes=2,' in text
    proof=dict(status='passed',checks=['two real-data head optimization steps','nonhead parameters bitwise unchanged',
        'swing and later-action head rows exactly retained','strict checkpoint reload','short diagnostic launches 2000-step budget'],temporary_weights_and_metrics_removed=True)
    output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();check(args.output)
