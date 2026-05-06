import asyncio, json, websockets

async def test_hf():
    url = 'wss://asabhishek-polaris-v3.hf.space/ws'
    print(f"Connecting to {url}...")
    try:
        async with websockets.connect(url, open_timeout=15) as ws:
            init = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            print(f"Connected! Init: tasks={init.get('tasks','?')}")
            
            await ws.send(json.dumps({'cmd': 'stop'}))
            await ws.send(json.dumps({'cmd': 'reset', 'seed': 42, 'task_id': 'environmental_recovery', 'chaos': 0.0}))
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({'cmd': 'start', 'seed': 42, 'chaos': 0.0, 'task_id': 'environmental_recovery'}))
            await ws.send(json.dumps({'cmd': 'speed', 'value': 50}))
            
            print("Simulation started, waiting for results...")
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if msg.get('type') == 'step':
                    s = msg.get('step', 0)
                    if s % 100 == 0:
                        print(f"  Step {s}...")
                elif msg.get('type') == 'episode_end':
                    outcome = 'COLLAPSED' if msg.get('collapsed') else 'SURVIVED'
                    print(f"\nHF RESULT: steps={msg['steps']} score={msg['score']:.3f} {outcome}")
                    print("HF SIMULATION WORKS!")
                    break
                elif msg.get('type') == 'error':
                    print(f"ERROR: {msg.get('message')}")
                    break
    except Exception as e:
        print(f"FAILED: {e}")

asyncio.run(test_hf())
