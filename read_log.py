import json
d = json.load(open('outputs/training_v4/training_log.json'))
print("EVALS:")
for e in d['evals']:
    print(f"  step={e['step']} {e['task']}: score={e['score']} collapse={e['collapse_rate']} coop={e.get('cooperation','?')} [{e['phase']}]")
steps = d['steps']
print(f"\nTRAINING: {len(steps)} steps completed")
if steps:
    print(f"Last: step={steps[-1]['step']} loss={steps[-1]['loss']} best={steps[-1]['best_score']} coop={steps[-1]['cooperation']}")
    # Show every 10th step
    print("\nProgress (every 10th):")
    for s in steps:
        if s['step'] % 10 == 0:
            print(f"  step={s['step']} loss={s['loss']:.4f} best={s['best_score']:.4f} coop={s['cooperation']:.2f} {'DEAD' if s['collapsed'] else 'ALIVE'}")
