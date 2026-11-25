"""Repeated inference on identical real observations with episode memory reset."""
import argparse,json
from pathlib import Path


def main(output):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from excavator_vlm.policies.act.inference import load_act
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
    checkpoint=root/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
    data=root/'outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset'
    policy,pre,_=load_act(checkpoint,'cuda:0',original_act=True)
    dataset=LeRobotDataset(repo_id='local/repeat',root=data,episodes=[112],video_backend='pyav')
    batches=[]
    for index in (0,3,6):
        row=dataset[index]
        if int(row['episode_index'])!=112:raise ValueError('Unexpected diagnostic observation parent')
        batches.append(pre({key:row[key][None] for key in ('observation.state',*policy.config.image_features)}))
    predictions=[]
    with torch.inference_mode():
        for repeat in range(5):
            policy.reset()
            predictions.append(torch.stack([policy.predict_action_chunk(dict(batch)) for batch in batches]).cpu())
    deviations=[(value-predictions[0]).abs().max().item() for value in predictions[1:]]
    result=dict(status='completed',observations_parent=112,frames=[0,3,6],repetitions=5,
        memory_reset_each_repetition=True,identical_observations=True,checkpoint=str(checkpoint),
        max_normalized_action_difference=max(deviations),exact_equal=all(torch.equal(value,predictions[0]) for value in predictions),
        scope='Determinism check for a fixed three-observation sequence; does not establish full simulator trajectory reproducibility')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
