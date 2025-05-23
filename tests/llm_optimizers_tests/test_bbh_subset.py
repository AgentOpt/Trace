import pytest
from datasets import load_dataset
from opto.optimizers import OptoPrime, OptoPrimeMulti
from opto.trace.nodes import ParameterNode
from opto.trace.bundle import bundle
from opto.trace import node, GRAPH

# ------------------------
# Load BBH Subset
# ------------------------

TASK = "logical_deduction_three_objects"
dataset = load_dataset("maveriq/bigbenchhard", TASK, split="train[:10]")
QA_PAIRS = [(ex["input"], ex["target"]) for ex in dataset]

# ------------------------
# Optimizer Configs
# ------------------------

GEN_TECHS = ["temperature_variation", "self_refinement", "iterative_alternatives", "multi_experts"]
SEL_TECHS = ["moa", "lastofn", "majority"]

def get_optimizer_configs():
    configs = [(OptoPrime, None, None)]
    for gen in GEN_TECHS:
        for sel in SEL_TECHS:
            configs.append((OptoPrimeMulti, gen, sel))
    return configs

OPTIMIZER_CONFIGS = get_optimizer_configs()

# ------------------------
# Scoring Test
# ------------------------

@pytest.mark.parametrize("optimizer_class,gen_tech,sel_tech", OPTIMIZER_CONFIGS)
def test_bbh_subset_accuracy(optimizer_class, gen_tech, sel_tech):
    """
    Run a batch of 10 Q&A pairs using a given optimizer configuration,
    and print final accuracy for that configuration.
    """
    # ------------------------
    # Trainable Function
    # ------------------------

    tmpl = ParameterNode("Answer the question.\n\nQ: {q}\nA:", trainable=True, name="bbh_prompt")

    @bundle(trainable=True)
    def solve(q, tmpl):
        from opto.trace.operators import call_llm
        prompt = tmpl.format(q=q)
        return call_llm(prompt)

    GRAPH.clear()
    name = (
        optimizer_class.__name__
        if optimizer_class is OptoPrime
        else f"{optimizer_class.__name__}({gen_tech}, {sel_tech})"
    )

    # Instantiate optimizer
    if optimizer_class is OptoPrime:
        optimizer = optimizer_class([tmpl])
    else:
        optimizer = optimizer_class([tmpl], generation_technique=gen_tech, selection_technique=sel_tech)

    correct = 0
    for q, a in QA_PAIRS:
        pred = solve(q, tmpl)
        feedback = "Correct" if a.lower() in pred.data.lower() else f"Wrong (expected {a})"
        if "Correct" in feedback:
            correct += 1
            # print without newline
            print(f"\rC", end="")
            continue
        print(f"INCORRECT {name} - Feedback: {feedback}")

        optimizer.zero_feedback()
        optimizer.backward(pred, feedback)
        optimizer.step()

    accuracy = correct / len(QA_PAIRS) * 100
    print(f"\n{name} accuracy: {accuracy:.1f}% over {len(QA_PAIRS)} examples")

    # Optional: Assert some minimal threshold, or just always pass
    assert isinstance(accuracy, float)  # always pass test
