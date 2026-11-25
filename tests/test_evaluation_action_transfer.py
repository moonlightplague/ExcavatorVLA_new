"""Action-profile transfer is explicit and never loosens execution limits."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'ExcavatorVLA-vlm')]
from excavator_vlm.data.action_limits import action_contract
from evaluation_action_transfer import validate_action_transfer


def info(profile='excavator_velocity_v1'):
    return {'action_limit_profile':profile,'action_limits_rad_s':[6.,2.5,2.5,20. if 'slew20' in profile else 6.]}


class ActionTransferTests(unittest.TestCase):
    def test_matching_profile_needs_no_transfer(self):
        self.assertIsNone(validate_action_transfer({'action_contract':action_contract(info())},info()))

    def test_bucket20_to_bucket6_must_be_explicit_and_records_both_profiles(self):
        source={'action_contract':action_contract(info('excavator_bucket_slew20_v1'))}
        with self.assertRaisesRegex(ValueError,'different action-limit profiles'):
            validate_action_transfer(source,info())
        result=validate_action_transfer(source,info(),allow_tighter=True)
        self.assertEqual(result['checkpoint_action_contract']['limits_rad_s'][-1],20.)
        self.assertEqual(result['execution_action_contract']['limits_rad_s'][-1],6.)

    def test_transfer_cannot_loosen_limits_or_accept_forged_semantics(self):
        with self.assertRaisesRegex(ValueError,'only tighten'):
            validate_action_transfer({},info('excavator_bucket_slew20_v1'),allow_tighter=True)
        source=action_contract(info());source['units']='degrees/s'
        with self.assertRaises(ValueError):
            validate_action_transfer({'action_contract':source},info(),allow_tighter=True)


if __name__=='__main__':unittest.main()
