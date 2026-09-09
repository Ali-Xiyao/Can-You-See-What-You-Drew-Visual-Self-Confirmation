"""Sensitivity planning for a future independent selector validation cohort."""
from datetime import datetime, timezone
from math import ceil, sqrt
from statistics import NormalDist
from common import *


def main():
    # For a fixed pool baseline, selection-gain change is the difference of
    # two correctness indicators: {-1,0,1}. Variance is q - delta**2, where
    # q is the chance the two checkpoints differ in selection correctness.
    z=NormalDist().inv_cdf(.975)
    z_power=NormalDist().inv_cdf(.8)
    delta=.10
    halfwidth=.05
    rows=[]
    for q in (.10,.20,.50,1.0):
        n_width=ceil(z*z*q/halfwidth**2)
        n_power=ceil((z+z_power)**2*(q-delta**2)/delta**2)
        rows.append({'discordance_probability':q,'approx_n_for_95pct_halfwidth_5pp_near_zero_change':n_width,
                     'approx_n_for_80pct_power_at_10pp_change':n_power,
                     'approx_combined_n':max(n_width,n_power),
                     'approx_halfwidth_at_200_scenes':z*sqrt(q/200)})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_arm':'naive',
            'estimand':'Change in original deterministic selection gain on identical independent candidate pools; one pool per independent scene.',
            'target_change':delta,'confidence':.95,'target_halfwidth':halfwidth,'target_power':.8,
            'scenarios':rows,
            'limitations':['Normal approximations for planning only, not assurance of achieved power or coverage.',
                           'q is not fixed to a sparse pilot point estimate; multiple scenarios expose sensitivity.',
                           'Label uncertainty, clustering, scene sampling, repeated looks and training-seed variation require separate allowance.',
                           'These numbers do not address the generative D* 2pp margin and do not justify an automatic 800-scene choice.',
                           '200 pools times 8 candidates is 1600 images. A 10% unresolved assumption gives 160 referrals, not a guaranteed or maximum human workload.',
                           'A fixed-image validation set measures selector drift. Generative outcome evaluation requires checkpoint-specific generated images.'],
            'new_image_generation_authorized_by_this_file':False}
    save_new(SIDE/'precision-planning.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
