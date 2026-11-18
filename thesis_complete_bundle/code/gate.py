"""Pre-specified evidence gate. Validation selects; locked test comparison is separate."""
import json, random
from pathlib import Path

def evaluate_gate(results, output, minimum_delta=.01, alpha=.05, bootstrap_samples=5000, seed=2026):
    """Paired percentile bootstrap on accuracy deltas vs strongest named validation control.

    PASS requires lower 95% bound and point improvement above the prespecified practical
    threshold. Test predictions are neither read nor accepted by this function.
    """
    val=results.get('validation',{})
    dynamic=val.get('ordered_predictions'); labels=val.get('labels')
    controls=val.get('control_predictions',[])
    if not bootstrap_samples or not 0 < alpha < 1 or minimum_delta < 0:
        raise ValueError('bootstrap_samples>0, alpha in (0,1), and nonnegative minimum_delta required')
    ids=val.get('ids')
    if dynamic is None or labels is None or ids is None or not controls or not len(labels):
        decision={'decision':'INCONCLUSIVE','reason':'validation IDs, labels, ordered predictions, and at least one matched control are required','minimum_accuracy_delta':minimum_delta,'alpha':alpha}
    elif len(dynamic)!=len(labels) or len(ids)!=len(labels) or len(set(ids))!=len(ids) or any(len(x)!=len(labels) for x in controls):
        decision={'decision':'INCONCLUSIVE','reason':'prediction arrays do not align with validation labels','minimum_accuracy_delta':minimum_delta,'alpha':alpha}
    elif any(int(y) not in range(5) for y in labels) or any(int(y) not in range(5) for y in dynamic) or any(int(p) not in range(5) for x in controls for p in x):
        decision={'decision':'INCONCLUSIVE','reason':'validation grades and predictions must be integers 0..4','minimum_accuracy_delta':minimum_delta,'alpha':alpha}
    else:
        correct=[[int(pred[i])==int(labels[i]) for pred in controls] for i in range(len(labels))]
        control_acc=[sum(row[j] for row in correct)/len(labels) for j in range(len(controls))]
        best=max(range(len(controls)),key=lambda j:control_acc[j])
        delta=[(1.0 if int(dynamic[i])==int(labels[i]) else 0.0)-(1.0 if correct[i][best] else 0.0) for i in range(len(labels))]
        rng=random.Random(seed); boots=[]
        for _ in range(bootstrap_samples):
            boots.append(sum(delta[rng.randrange(len(delta))] for _ in delta)/len(delta))
        boots.sort(); lo=boots[max(0,int(alpha/2*bootstrap_samples))]; hi=boots[min(bootstrap_samples-1,int((1-alpha/2)*bootstrap_samples))]; point=sum(delta)/len(delta)
        state='PASS' if point>minimum_delta and lo>minimum_delta else ('FAIL' if hi<=minimum_delta else 'INCONCLUSIVE')
        decision={'decision':state,'reason':'paired percentile bootstrap versus highest-accuracy named validation control','minimum_accuracy_delta':minimum_delta,'alpha':alpha,'point_delta_accuracy':point,'bootstrap_ci':[lo,hi],'samples':len(labels),'controls':len(controls),'strongest_control_index':best,'control_accuracies':control_acc,'seed':seed,'bootstrap_samples':bootstrap_samples}
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(decision,indent=2)+'\n')
    return decision

def qrc_is_eligible(gate_path): return json.loads(Path(gate_path).read_text()).get('decision')=='PASS'
