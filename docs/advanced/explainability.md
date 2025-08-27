# Enforcement & Explainability

Let's dive into how Parlant enforces the conversation model consistently and provides visibility into your agent's situational understanding and decision-making process.

In this section, you'll learn:

1. How _Attentive Reasoning Queries (ARQs)_ enforce the conversation model
1. How to use ARQ artifacts to troubleshoot and improve behavior

### Understanding Runtime Enforcement

During message generation, Parlant ensures guidelines are followed consistently in real-time conversations through our [Attentive Reasoning Queries](https://arxiv.org/abs/2503.03669#:~:text=We%20present%20Attentive%20Reasoning%20Queries%20%28ARQs%29%2C%20a%20novel,in%20Large%20Language%20Models%20through%20domain-specialized%20reasoning%20blueprints.) prompting method. Rather than simply adding guidelines to prompts and hoping for the best, Parlant employes explicit techniques to maximize the LLM's ability and likelihood to adhere to your guidelines.

Attentive Reasoning Queries (ARQs) are essentially structured reasoning blueprints built into prompts that guide LLMs through specific thinking patterns when making decisions or solving problems. Rather than hoping an AI agent naturally considers all important factors, ARQs explicitly outline reasoning steps for different domains—like having specialized mental checklists to go through.

What makes ARQs effective for behavioral enforcement is that they force attention on critical considerations that might otherwise be overlooked. The model must work through predetermined reasoning stages (like context assessment, solution exploration, critique, and decision formation), ensuring it consistently evaluates important constraints before taking action.

![ARQs](https://arxiv.org/html/2503.03669v1/x1.png)

**Figure:** Illustration of ARQs (taken from the [research paper](https://arxiv.org/abs/2503.03669#:~:text=We%20present%20Attentive%20Reasoning%20Queries%20%28ARQs%29%2C%20a%20novel,in%20Large%20Language%20Models%20through%20domain-specialized%20reasoning%20blueprints.))

Besides increasing accuracy and conformance to instructions, this process creates, as a byproduct, transparent, auditable reasoning paths that help maintain alignment with desired behaviors.

ARQs are flexible enough to adapt to different contexts and risk levels, with reasoning blueprints that can be tailored to specific domains or regulatory requirements. While there's some computational overhead to this more deliberate thinking process, carefully designed ARQs can beat Chain-of-Thought reasoning in both accuracy and latency.

Parlant uses different sets of ARQs for each of its components (e.g., guideline matching, tool-calling, or message composition), and dynamically specializes the ARQs to the specific entity it's evaluating, whether it's a particular guideline, tool, or conversational context.

Here's an illustrated example from the `GuidelineMatcher`'s logs:

```json
{
  "guideline_id": "fl00LGUyZX",
  "condition": "the customer wants to return an item",
  "condition_application_rationale": "The customer explicitly stated that they need to return a sweater that doesn't fit, indicating a desire to return an item.",
  "condition_applies": true,
  "action": "get the order number and item name and them help them return it",
  "action_application_rationale": [
    {
      "action_segment": "Get the order number and item name",
      "rationale": "I've yet to get the order number and item name from the customer."
    },
    {
      "action_segment": "Help them return it",
      "rationale": "I've yet to offer to help the customer in returning the item."
    }
  ],
  "applies_score": 9
}
```

### Explaining and Troubleshooting Agent Behavior
Message generation in Parlant goes through quite a lot of quality assurance. As mentioned above, ARQs produce artifacts that can help explain how the agent interpreted circumstances and instructions.

When you run into issues, you can inspect these artifacts to better understand why the agent responded the way it did, and whether it correctly interpreted your intentions.

Over time, this feedback loop helps you build more precise and effective sets of guidelines.

![Explainability in Parlant](https://parlant.io/img/explainability.gif)
