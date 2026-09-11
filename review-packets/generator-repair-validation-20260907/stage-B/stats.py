"""Scene bootstrap and worst-completion bounds, fixed before B data."""
import numpy as np

def draws(n):
    assert n > 0
    return np.random.default_rng(20260907).integers(n, size=(5000,n))

def interval(values):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    d = x[draws(len(x))].mean(1)
    return {'n_scenes':len(x), 'mean':float(x.mean()),
            'ci95':np.quantile(d,[.025,.975]).tolist(),
            'ci90':np.quantile(d,[.05,.95]).tolist(),
            'lower_one_sided95':float(np.quantile(d,.05))}

def bounded_summary(bounds):
    x = np.asarray(bounds, dtype=float)
    assert x.shape == (len(x),2) and len(x) and np.isfinite(x).all()
    assert np.all(x[:,0] <= x[:,1] + 1e-12)
    d = x[draws(len(x))].mean(1)
    identified = bool(np.allclose(x[:,0], x[:,1], rtol=0, atol=1e-12))
    return {'n_scenes':len(x), 'mean_bounds':x.mean(0).tolist(),
            'ci95_worst_completion':[float(np.quantile(d[:,0],.025)),float(np.quantile(d[:,1],.975))],
            'ci90_worst_completion':[float(np.quantile(d[:,0],.05)),float(np.quantile(d[:,1],.95))],
            'lower_one_sided95_worst_completion':float(np.quantile(d[:,0],.05)),
            'point_identified':identified, 'point':interval(x[:,0]) if identified else None}

def weighted_bounds(weights, labels):
    assert len(weights) == len(labels)
    low = high = 0.
    for w, y in zip(weights, labels):
        if y is None: low += min(0.,w); high += max(0.,w)
        else:
            assert y in (0,1)
            low += w*y; high += w*y
    return [low,high]

def top_weights(scores):
    s = np.asarray(scores)
    hits = s == s.max()
    return hits.astype(float)/hits.sum()

def paired_selection(w_new, w_base, labels):
    return weighted_bounds(np.asarray(w_new)-np.asarray(w_base),labels)

def ability_summary(rows):
    """Rows contain known group differences and all unresolved family facts.

    Unknown truth can change group membership. For a scene with n known group
    facts and m unknown family facts, allow all m to enter with differences -1
    or +1. For unknown-only scenes allow them all to enter at the adverse
    extreme. These are deliberately conservative bounds, not imputed labels.
    """
    known = [r for r in rows if r['differences']]
    instances = sum(len(r['differences']) for r in known)
    observed = interval([float(np.mean(r['differences'])) for r in known]) if known else None
    lows=[]; highs=[]; eligible=[]
    for r in rows:
        n=len(r['differences']); m=r['unknown_family']; s=sum(r['differences'])
        assert m >= 0 and all(-1 <= d <= 1 for d in r['differences'])
        if n:
            lows.append((s-m)/(n+m)); highs.append((s+m)/(n+m)); eligible.append(1)
        elif m:
            lows.append(-1.); highs.append(1.); eligible.append(1)
        else:
            lows.append(0.); highs.append(0.); eligible.append(0)
    lo=np.array(lows); hi=np.array(highs); e=np.array(eligible)
    idx=draws(len(rows)); den=e[idx].sum(1)
    dl=np.divide(lo[idx].sum(1),den,out=np.full(5000,-1.),where=den>0)
    dh=np.divide(hi[idx].sum(1),den,out=np.full(5000,1.),where=den>0)
    lower=float(np.quantile(dl,.05)); upper=float(np.quantile(dh,.95))
    enough=instances >= 150 and len(known) >= 50
    unknown=sum(r['unknown_family'] for r in rows)
    return {'known_instances':instances, 'known_scenes':len(known),
            'unknown_family_instances':unknown, 'known_only':observed,
            'mean_bounds':[float(lo.sum()/e.sum()),float(hi.sum()/e.sum())] if e.sum() else [-1.,1.],
            'ci90_worst_completion':[lower,upper], 'lower_one_sided95_worst_completion':lower,
            'enough_instances_and_scenes':enough,
            'retained':bool(enough and observed['lower_one_sided95'] >= -.10 and lower >= -.10)}

def retention_decision(ability_ok, summary):
    lo,hi=summary['ci90_worst_completion']
    if summary['ci95_worst_completion'][1] < -.05: return 'residual_decline'
    if ability_ok and lo >= -.05 and hi <= .05: return 'retained_equivalent'
    return 'inconclusive'

def context_decision(on_random, blank_random, blank_on):
    lo,hi=on_random['ci90_worst_completion']
    return bool(lo >= -.05 and hi <= .05 and
                blank_random['ci95_worst_completion'][0] >= .15 and
                blank_on['ci95_worst_completion'][0] >= .10)
