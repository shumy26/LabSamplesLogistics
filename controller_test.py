import json
import os
import random
import sys
from collections import Counter


def export_trace_to_dot(trace, dot_filename):
  with open(dot_filename, "w", encoding="utf-8") as f:
    f.write("digraph Trace {\n")
    f.write("  rankdir=LR;\n")
    f.write('  bgcolor="transparent";\n')
    f.write('  node [shape=record, style=filled, fontname="Helvetica"];\n')
    f.write('  edge [fontname="Helvetica", fontsize=10, color="#546e7a"];\n\n')

    for item in trace:
      step = item["step"]
      node_id = item["node"]
      is_goal = item.get("is_goal", False)
      goal_type = item.get("goal_type", "")

      in_actives = [key for key, value in item["inputs"].items() if value == 1]
      out_actives = [key for key, value in item["outputs"].items() if value == 1]

      in_str = ", ".join(in_actives) if in_actives else "none"
      out_str = ", ".join(out_actives) if out_actives else "none"

      label = (
          f"Step {step} (Node {node_id}) | {{ IN: {in_str} | OUT: {out_str} }}"
      )

      if is_goal:
        fillcolor = "#fff59d" if "HUMAN" in goal_type else "#c8e6c9"
      elif step == 0:
        fillcolor = "#e1f5fe"
      else:
        fillcolor = "#ffffff"

      f.write(f'  step_{step} [label="{label}", fillcolor="{fillcolor}"];\n')

    f.write("\n")
    for i in range(len(trace) - 1):
      src_step = trace[i]["step"]
      dst_step = trace[i + 1]["step"]
      next_in_actives = [
          key for key, value in trace[i + 1]["inputs"].items() if value == 1
      ]
      edge_label = ", ".join(next_in_actives) if next_in_actives else "---"
      f.write(
          f'  step_{src_step} -> step_{dst_step} [label=" in: {edge_label}'
          ' "];\n'
      )

    f.write("}\n")


def simulate_environment_step(
    step, current_env, current_sys, params, rng, tracker
):
  next_env = current_env.copy()
  events_triggered = []

  arriving_at_lab = False
  if current_sys.get("goto_lab") == 1 and current_env.get("at_lab") == 0:
    next_env["at_floor"] = 0
    next_env["at_lab"] = 1
    arriving_at_lab = True
    events_triggered.append("Robot arrived at lab")
  elif current_sys.get("goto_floor") == 1 and current_env.get("at_floor") == 0:
    next_env["at_lab"] = 0
    next_env["at_floor"] = 1
    events_triggered.append("Robot returned to patient floor")

  prob_scan_fail = params.get("prob_scan_fail", 0.0)
  if (
      current_sys.get("scan") == 1
      and current_env.get("at_floor") == 1
      and current_env.get("auth_present") == 1
  ):
    if rng.random() < prob_scan_fail:
      next_env["barcode_ok"] = 0
      tracker["scan_fails"] += 1
      events_triggered.append(
          f"Failure to scan ({tracker['scan_fails']}x consecutive)"
      )
    else:
      next_env["barcode_ok"] = 1
      tracker["scan_fails"] = 0
      events_triggered.append("Barcode successfully validated")

  if (
      current_env.get("barcode_ok") == 1
      and current_sys.get("load_machine") == 0
      and current_env.get("human_pickup") == 0
  ):
    next_env["barcode_ok"] = 1

  max_scan_fails = params.get("max_scan_fails_before_rescue", 3)
  max_idle = params.get("max_idle_before_rescue", 5)

  rescue_needed = False
  if tracker["scan_fails"] >= max_scan_fails and next_env.get("at_floor") == 1:
    rescue_needed = True
    events_triggered.append(
        "HUMAN_PICKUP: human intervention needed"
    )
  elif (
      tracker["idle_steps"] >= max_idle
      and next_env.get("request") == 1
      and next_env.get("barcode_ok") == 0
  ):
    rescue_needed = True
    events_triggered.append(
        "HUMAN_PICKUP: failure to get authorized personnel"
    )

  prob_human_pickup_lab = params.get("prob_human_pickup_lab", 0.0)
  pickup_at_lab = False
  if (
      next_env.get("at_lab") == 1 or arriving_at_lab
  ) and next_env.get("barcode_ok") == 1:
    if rng.random() < prob_human_pickup_lab:
      pickup_at_lab = True
      events_triggered.append(
          "HUMAN_PICKUP no Lab: spontaneous human intervention"
      )

  next_env["human_pickup"] = 1 if (rescue_needed or pickup_at_lab) else 0

  sample_consumed = (current_sys.get("load_machine") == 1) or (
      current_env.get("human_pickup") == 1
  )
  if sample_consumed:
    next_env["barcode_ok"] = 0
    next_env["human_pickup"] = 0
    next_env["request"] = 0
    events_triggered.append("Sample loaded")
  else:
    req_step = params.get("request_at_step", 0)
    if step >= req_step:
      next_env["request"] = 1

  if params.get("has_auth", True) and next_env.get("request") == 1:
    auth_step = params.get("auth_at_step", 1)
    auth_duration = params.get("auth_duration", 4)
    next_env["auth_present"] = (
        1 if (auth_step <= step < auth_step + auth_duration) else 0
    )
  else:
    next_env["auth_present"] = 0

  if (
      next_env.get("request") == 1
      and next_env.get("auth_present") == 0
      and current_sys.get("goto_lab") == 0
      and next_env.get("barcode_ok") == 0
  ):
    tracker["idle_steps"] += 1
  else:
    tracker["idle_steps"] = 0

  return next_env, events_triggered


def run_single_mission(
    mission_config, nodes, variables, num_inputs=6, max_steps=20
):
  env_vars = variables[:num_inputs]
  sys_vars = variables[num_inputs:]

  seed = mission_config.get("seed", random.randint(1, 999999))
  rng = random.Random(seed)

  tracker = {"scan_fails": 0, "idle_steps": 0}

  current_state_id = "0"
  trace = []
  status = "RUNNING"
  reason = ""
  history_events = []

  for step in range(max_steps):
    node_info = nodes.get(current_state_id)
    if not node_info:
      status = "FAILED"
      reason = f"Node {current_state_id} does not exist"
      break

    state_vector = node_info["state"]
    env_dict = dict(zip(env_vars, state_vector[:num_inputs]))
    sys_dict = dict(zip(sys_vars, state_vector[num_inputs:]))

    is_machine_goal = (
        env_dict.get("at_lab") == 1 and sys_dict.get("load_machine") == 1
    )
    is_lab_pickup = (
        env_dict.get("at_lab") == 1 and env_dict.get("human_pickup") == 1
    )
    is_floor_rescue = (
        env_dict.get("at_floor") == 1 and env_dict.get("human_pickup") == 1
    )

    reached_goal = is_machine_goal or is_lab_pickup or is_floor_rescue

    goal_type = ""
    if is_machine_goal:
      goal_type = "MACHINE_DELIVERY"
    elif is_lab_pickup:
      goal_type = "HUMAN_PICKUP_LAB"
    elif is_floor_rescue:
      goal_type = "HUMAN_RESCUE_FLOOR"

    trace.append({
        "step": step,
        "node": current_state_id,
        "inputs": env_dict,
        "outputs": sys_dict,
        "is_goal": reached_goal,
        "goal_type": goal_type,
    })

    if reached_goal:
      if is_floor_rescue:
        status = "RECOVERED"
        reason = (
            "Human intervention solved bad situation at patient floors"
        )
      elif is_lab_pickup:
        status = "SUCCESS"
        reason = (
            "Human picked up the sample and completed the mission"
        )
      else:
        status = "SUCCESS"
        reason = (
                "Automatic machine loading"
        )
      break

    transitions = [str(t) for t in node_info.get("trans", [])]
    if not transitions:
      status = "FAILED"
      reason = f"Reached a sink state (no valid transitions)"
      break

    next_env_dict, step_events = simulate_environment_step(
        step, env_dict, sys_dict, mission_config["params"], rng, tracker
    )
    if step_events:
      history_events.extend([f"[S{step}] {ev}" for ev in step_events])

    next_env_inputs = [next_env_dict[v] for v in env_vars]

    next_state_id = None
    for succ_id in transitions:
      if succ_id in nodes:
        if nodes[succ_id]["state"][:num_inputs] == next_env_inputs:
          next_state_id = succ_id
          break

    if next_state_id is None:
      status = "FAILED"
      reason = (
              f"Assumption violation on step {step} with inputs: {next_env_inputs}"
      )
      break

    current_state_id = next_state_id
  else:
    if status == "RUNNING":
      status = "FAILED"
      reason = f"Timeout at {max_steps} steps"

  return {
      "mission_id": mission_config["id"],
      "category": mission_config.get("category", "General"),
      "description": mission_config["description"],
      "seed": seed,
      "parameters": mission_config["params"],
      "events": history_events,
      "status": status,
      "reason": reason,
      "total_steps": len(trace),
      "trace": trace,
  }


def run_single_example_mission(
    json_file="controller.json", output_dir="sim_example"
):
  os.makedirs(output_dir, exist_ok=True)

  try:
    with open(json_file, "r", encoding="utf-8") as f:
      content = f.read().strip()
      if not content:
        raise ValueError(
            f"the file '{json_file}' is empty. Recompile using slugs"
        )
      data = json.loads(content)
  except Exception as e:
    print(f"\n[ERROR] Failed to load '{json_file}': {e}")
    return

  variables = data.get("variables", [])
  nodes = data.get("nodes", {})

  nominal_config = {
      "id": "EXAMPLE_MISSION",
      "category": "Example_Machine",
      "description": (
          "Perfect example mission: correct scanning and unloading"
      ),
      "seed": 12345,
      "params": {
          "request_at_step": 0,
          "auth_at_step": 1,
          "auth_duration": 6,
          "has_auth": True,
          "prob_scan_fail": 0.0,
          "prob_human_pickup_lab": 0.0,
          "max_scan_fails_before_rescue": 3,
          "max_idle_before_rescue": 6,
      },
  }

  print("\n" + "=" * 70)
  print(f" EXAMPLE MISSION EXECUTION")
  print("=" * 70)

  result = run_single_mission(nominal_config, nodes, variables, num_inputs=6)

  print(
      f"\n{'Step':<6} | {'Node':<6} | {'Active inputs':<26} |"
      " {'Active outputs'}"
  )
  print("-" * 70)
  for item in result["trace"]:
    in_actives = [k for k, v in item["inputs"].items() if v == 1]
    out_actives = [k for k, v in item["outputs"].items() if v == 1]
    in_str = ", ".join(in_actives) if in_actives else "none"
    out_str = ", ".join(out_actives) if out_actives else "none"
    print(
        f"P{item['step']:<5} | Node {item['node']:<2} | {in_str:<26} |"
        f" {out_str}"
    )

  print("-" * 70)
  print(f"Final Status : [{result['status']}]")
  print(f"Reason       : {result['reason']}")
  print(f"Total Steps  : {result['total_steps']}")

  if result["events"]:
    print("\nEvent History:")
    for ev in result["events"]:
      print(f"  • {ev}")

  dot_path = os.path.join(output_dir, f"{nominal_config['id']}.dot")
  export_trace_to_dot(result["trace"], dot_path)
  print(f"\nDOT file successfully written to: '{dot_path}'\n")


def generate_missions():
  missions = []
  idx = 1

  for _ in range(50):
    req = random.randint(0, 2)
    auth = req + random.randint(1, 2)
    missions.append({
        "id": f"SIM_{idx:03d}_ScanFault",
        "category": "ScanFault",
        "description": (
            "Broken scanner. barcode_ok never activates"
        ),
        "seed": random.randint(1, 999999),
        "params": {
            "request_at_step": req,
            "auth_at_step": auth,
            "auth_duration": random.randint(3, 6),
            "has_auth": True,
            "prob_scan_fail": 1.0,
            "prob_human_pickup_lab": 0.0,
            "max_scan_fails_before_rescue": random.randint(1, 3),
            "max_idle_before_rescue": random.randint(4, 6),
        },
    })
    idx += 1

  for _ in range(50):
    req = random.randint(0, 3)
    missions.append({
        "id": f"SIM_{idx:03d}_MissingAuthRescue",
        "category": "AbandonedRequest",
        "description": (
            "No authorized personnel available"
        ),
        "seed": random.randint(1, 999999),
        "params": {
            "request_at_step": req,
            "has_auth": False,
            "prob_scan_fail": 0.0,
            "prob_human_pickup_lab": 0.0,
            "max_scan_fails_before_rescue": 2,
            "max_idle_before_rescue": random.randint(3, 5),
        },
    })
    idx += 1

  for _ in range(50):
    req = random.randint(0, 3)
    auth = req + random.randint(1, 2)
    missions.append({
        "id": f"SIM_{idx:03d}_CorrectDelivery",
        "category": "NoAdversity",
        "description": (
            "Normal event flow, no adversity"
        ),
        "seed": random.randint(1, 999999),
        "params": {
            "request_at_step": req,
            "auth_at_step": auth,
            "auth_duration": random.randint(4, 7),
            "has_auth": True,
            "prob_scan_fail": 0.0,
            "prob_human_pickup_lab": 0.0,
            "max_scan_fails_before_rescue": 3,
            "max_idle_before_rescue": 6,
        },
    })
    idx += 1

  for _ in range(50):
    req = random.randint(0, 3)
    auth = req + random.randint(1, 2)
    missions.append({
        "id": f"SIM_{idx:03d}_LabHumanPickup",
        "category": "LabPickup",
        "description": (
            "A human always intervenes in the lab"
        ),
        "seed": random.randint(1, 999999),
        "params": {
            "request_at_step": req,
            "auth_at_step": auth,
            "auth_duration": random.randint(4, 7),
            "has_auth": True,
            "prob_scan_fail": 0.0,
            "prob_human_pickup_lab": 1.0,
            "max_scan_fails_before_rescue": 3,
            "max_idle_before_rescue": 6,
        },
    })
    idx += 1

  for _ in range(50):
    req = random.randint(0, 2)
    auth = req + 1
    scan_fail = round(random.uniform(0.10, 0.40), 2)
    human_intervention_lab = round(random.uniform(0.20, 0.70), 2)
    missions.append({
        "id": f"SIM_{idx:03d}_RandomStress",
        "category": "RandomStress",
        "description": (
            f"Adverse environment (ScanFail: {int(scan_fail*100)}%, LabPickup:"
            f" {int(human_intervention_lab*100)}%)."
        ),
        "seed": random.randint(1, 999999),
        "params": {
            "request_at_step": req,
            "auth_at_step": auth,
            "auth_duration": random.randint(4, 6),
            "has_auth": True,
            "prob_scan_fail": scan_fail,
            "prob_human_pickup_lab": human_intervention_lab,
            "max_scan_fails_before_rescue": 2,
            "max_idle_before_rescue": 5,
        },
    })
    idx += 1

  return missions


def run_batch_simulation(
    json_file="controller.json", output_dir="sim_results_massive"
):
  os.makedirs(output_dir, exist_ok=True)

  try:
    with open(json_file, "r", encoding="utf-8") as f:
      content = f.read().strip()
      if not content:
        raise ValueError(
            f"The file '{json_file}' is empty. Recompile using slugs"
        )
      data = json.loads(content)
  except Exception as e:
    print(f"\n[ERROR] Failure to load '{json_file}': {e}")
    return

  variables = data.get("variables", [])
  nodes = data.get("nodes", {})

  missions = generate_missions()
  all_results = []

  print("\n" + "=" * 65)
  print(f" SIMULATION BATCH ({len(missions)} SCENARIOS / 5 SITUATIONS)")
  print("=" * 65 + "\n")

  for i, m in enumerate(missions, 1):
    result = run_single_mission(m, nodes, variables, num_inputs=6)
    all_results.append(result)

    if i <= 10 or result["status"] in ["FAILED", "RECOVERED"]:
      dot_path = os.path.join(output_dir, f"{result['mission_id']}.dot")
      export_trace_to_dot(result["trace"], dot_path)

    print(
        f"[{i:03d}/{len(missions):03d}]"
        f" [{result['status']:9s}] {result['mission_id']:<28s} |"
        f" {result['reason']}"
    )

  total = len(all_results)
  successes = sum(1 for r in all_results if r["status"] == "SUCCESS")
  recovered = sum(1 for r in all_results if r["status"] == "RECOVERED")
  failures = sum(1 for r in all_results if r["status"] == "FAILED")

  categories = set(r["category"] for r in all_results)
  reasons = Counter(r["reason"] for r in all_results)

  print("\n" + "=" * 65)
  print("                         RESULTS PANEL               ")
  print("=" * 65)
  print(f" Total simulations         : {total}")
  print(
      f" Nominal Sucesses          : {successes} ({(successes/total)*100:.1f}%)"
  )
  print(
      " Rescued                   : "
      f"{recovered} ({(recovered/total)*100:.1f}%)"
  )
  print(
      f" Failures / Violations     : {failures} ({(failures/total)*100:.1f}%)"
  )
  print(
      " Global Success Rate       :"
      f" {((successes+recovered)/total)*100:.1f}%\n"
  )

  print("--- Performance by Category (50 executions each) ---")
  for cat in sorted(categories):
    cat_results = [r for r in all_results if r["category"] == cat]
    cat_tot = len(cat_results)
    cat_ok = sum(1 for r in cat_results if r["status"] in ["SUCCESS", "RECOVERED"])
    print(
        f" • {cat:<32s}: {cat_ok:2d}/{cat_tot:2d} solved"
        f" ({(cat_ok/cat_tot)*100:5.1f}%)"
    )

  print("\n--- Statistics ---")
  for r_text, count in reasons.most_common():
    print(f" [{count:3d}x] {r_text}")

  summary_path = os.path.join(output_dir, "test_results.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

  print(
      f"\nResults saved to: '{summary_path}' and DOT files at"
      f" '{output_dir}/'"
  )


if __name__ == "__main__":
  print("\n" + "=" * 50)
  print(" REACTIVE CONTROLLER SIMULATION (SLUGS)")
  print("=" * 50)
  print(" [1] Execute 250 different missions with probabilistic parameters")
  print(" [2] Execute a single example mission in which everything goes right")
  print("=" * 50)

  op = input("Select an option (1 or 2): ").strip()
  if op == "1":
    run_batch_simulation("controller.json", "sim_results")
  elif op == "2":
    run_single_example_mission("controller.json", "sim_example")
  else:
    print("Invalid option")
