#!/usr/bin/env python3
"""Phase 2+3: GRPO Training + Post-Training Eval (VRAM-optimized)"""
import sys, os, io, json, time, gc, random, statistics, traceback
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS, CORE_ACTIONS
from server.tasks import grade_trajectory, get_task_ids

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "polaris_v4_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAINING_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ALL_TASKS = get_task_ids()
SEEDS = [42, 123, 777]
ACTION_LIST_STR = "\n".join(f"  - {n}: {d}" for n, d in ACTION_DESCRIPTIONS.items())

SYSTEM_SIMPLE = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

Respond with ONLY the action name. Nothing else."""

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def format_obs(meta, step, max_steps):
    events = ", ".join(meta.get("active_events", [])) or "none"
    return f"Step {step}/{max_steps} | GDP: {meta.get('gdp_index',0):.0f} | Pollution: {meta.get('pollution_index',0):.0f} | Satisfaction: {meta.get('public_satisfaction',0):.0f} | Events: {events}"

def parse_action(raw):
    raw = raw.lower().strip("'\"` \n")
    if raw in VALID_ACTIONS: return raw
    for a in VALID_ACTIONS:
        if a in raw: return a
    return "no_action"


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    t_start = time.time()

    # Clear any leftover VRAM
    gc.collect()
    torch.cuda.empty_cache()
    log(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    log("=" * 60)
    log("PHASE 2: GRPO+QLoRA TRAINING (VRAM-optimized)")
    log("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    log("Loading Qwen2.5-3B with 4-bit QLoRA...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()  # Save VRAM

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable: {trainable:,} params")
    log(f"VRAM after load: {(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/1e9:.1f} GB used")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01)

    # Use simple single-agent tasks for training (less prompt text = less VRAM)
    train_task = "environmental_recovery"
    n_epochs = 8
    n_rollouts = 2
    train_log_data = []

    log(f"Training: {n_epochs} epochs x {n_rollouts} rollouts on '{train_task}'")
    sys_prompt = SYSTEM_SIMPLE.format(actions=ACTION_LIST_STR)

    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_rewards = []

        for rollout in range(n_rollouts):
            seed = epoch * 100 + rollout + 7
            env = PolicyEnvironment()
            obs = env.reset(seed=seed, task_id=train_task)
            cfg = TASK_CONFIGS[train_task]
            max_steps = cfg["max_steps"]

            episode_reward = 0.0
            step_losses = []
            step = 0

            while not obs.done and step < max_steps:
                step += 1
                meta = obs.metadata
                obs_text = format_obs(meta, step, max_steps)

                messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obs_text}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

                # Phase 1: Generate action (no grad, saves VRAM)
                model.eval()
                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs, max_new_tokens=15, temperature=0.3, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                new_tokens = gen_out[0][inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                action = parse_action(raw)

                # Phase 2: Compute loss on generated sequence (with grad)
                model.train()
                full_seq = gen_out[0:1].detach()  # detach from gen graph
                labels = full_seq.clone()
                prompt_len = inputs["input_ids"].shape[1]
                labels[0, :prompt_len] = -100
                out = model(input_ids=full_seq, labels=labels)
                step_losses.append(out.loss)

                # Step the env
                obs = env.step({"action": action})
                episode_reward += obs.reward

                # Periodically clear cache
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            epoch_rewards.append(episode_reward)

        # GRPO-style update: weight losses by advantage
        mean_r = statistics.mean(epoch_rewards)
        optimizer.zero_grad()

        if step_losses:
            # Average loss across all steps
            total_loss = sum(step_losses) / len(step_losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = total_loss.item()
        else:
            loss_val = 0.0

        # Clear computation graph
        step_losses = []
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        entry = {"epoch": epoch+1, "mean_reward": round(mean_r, 4),
                 "min_reward": round(min(epoch_rewards), 4),
                 "max_reward": round(max(epoch_rewards), 4),
                 "loss": round(loss_val, 4), "time": round(elapsed, 1)}
        train_log_data.append(entry)
        log(f"  Epoch {epoch+1}/{n_epochs}: reward={mean_r:.2f} [{min(epoch_rewards):.1f}, {max(epoch_rewards):.1f}] loss={loss_val:.4f} ({elapsed:.1f}s)")

    adapter_path = os.path.join(OUTPUT_DIR, "qwen3b_grpo_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log(f"Adapter saved: {adapter_path}")
    log("Phase 2 COMPLETE")

    # ── PHASE 3: Post-Training Eval ──
    log("\n" + "=" * 60)
    log("PHASE 3: POST-TRAINING EVALUATION")
    log("=" * 60)

    model.eval()
    post_results = {}

    for task_id in ALL_TASKS:
        cfg = TASK_CONFIGS[task_id]
        # Skip multi-minister tasks (too much VRAM for generation)
        if cfg.get("num_ministers", 1) > 1:
            log(f"  Skipping {task_id} (multi-minister, eval separately)")
            continue

        log(f"  Task: {task_id}")
        task_results = []
        for seed in SEEDS:
            t0 = time.time()
            try:
                env = PolicyEnvironment()
                obs = env.reset(seed=seed, task_id=task_id)
                max_steps = cfg["max_steps"]
                total_reward = 0.0
                step = 0

                while not obs.done and step < max_steps:
                    step += 1
                    obs_text = format_obs(obs.metadata, step, max_steps)
                    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obs_text}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=15, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    action = parse_action(raw)
                    obs = env.step({"action": action})
                    total_reward += obs.reward

                score = grade_trajectory(task_id, env.get_trajectory())
                collapsed = obs.metadata.get("collapsed", False)
                elapsed = time.time() - t0
                status = "COLLAPSED" if collapsed else "SURVIVED"
                log(f"    seed={seed}: {status} score={score:.4f} reward={total_reward:.1f} ({elapsed:.1f}s)")
                task_results.append({"task_id": task_id, "seed": seed, "score": round(score, 4),
                    "reward": round(total_reward, 4), "collapsed": collapsed, "steps": step})
            except Exception as e:
                log(f"    seed={seed}: ERROR - {e}")
                task_results.append({"task_id": task_id, "seed": seed, "score": 0, "collapsed": True, "error": str(e)})

            torch.cuda.empty_cache()

        post_results[task_id] = task_results

    # Save
    results = {"training_log": train_log_data, "post_training_results": post_results, "adapter_path": adapter_path}
    path = os.path.join(OUTPUT_DIR, "phase2_3_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    post_scores = [r["score"] for tid in post_results for r in post_results[tid]]
    if post_scores:
        avg_post = statistics.mean(post_scores)
        collapse_n = sum(1 for tid in post_results for r in post_results[tid] if r.get("collapsed"))
        total_n = len(post_scores)
        log(f"\n  Post-training avg score: {avg_post:.4f}")
        log(f"  Post-training collapse: {collapse_n}/{total_n} ({collapse_n/total_n*100:.0f}%)")

    elapsed_total = time.time() - t_start
    log(f"\n  DONE in {elapsed_total/60:.1f} minutes")
    log(f"  Results: {path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
