# Relationships

Defining how **guidelines** and **journeys** relate to each other is a powerful (and advanced) part of behavior modeling.

#### Background & Motivation

Back in the day, our team was building a pizza-sales agent, which had the following guidelines (among others):
```python
offer_pepsi_instead_of_coke = await agent.create_guideline(
    condition="The customer wants a coke",
    action="Tell them we only have Pepsi",
)

handoff_if_upset = await agent.create_guideline(
    condition="The customer is becoming upset",
    action="Apologize and tell them you will transfer them to a manager",
    tools=[handoff_to_human_manager],
)
```

This initially worked well, until we encountered the following scenario:

> **Agent:** Do you want anything to drink with your order?
>
> **User:** A coke please
>
> **Agent:** I'm sorry, we only have Pepsi. Would you like that instead?
>
> **User:** Wait, what? I hate Pepsi. Why the hell don't you have coke?
>
> **Agent:** I'm sorry for this inconvenience. Let me transfer you to a manager. Meanwhile, can I offer you a Pepsi?
>
> **User:** Are you taking the piss out of me?

The agent's response was definitely not what we wanted, as it can come across as sarcastically hostile. But the poor AI agent was only following the guidelines we had given it, based on the conditions of the guidelines we had set.

And here's the thing. You will find that managing instructions is not just a technical challenge, it's also a human modeling challenge. We must consider how our instructions relate to each other to an automatic agent who's expected to take them quite literally. How they should relate to each other in different contexts, especially in more nuanced situations, is something that ultimately only we can decide.

In the case above, we wanted to ensure that the second guideline would be prioritized *over* the first one. We needed a priority relationship, which today, in Parlant, can be expressed quite simply, as follows:

```python
await handoff_if_upset.prioritize_over(offer_pepsi_instead_of_coke)
```

## Relationship Kinds

While these relationships might sound complex at first, they give you a ton of power as a modeler, making you much more capable of generating precise responses, consistently.

We recommend reviewing these relationships briefly to understand their purpose.

#### Relationship Types
These are the supported relationships. Each relationship is between a _source_ (notated **S**) and a _target_ (notated **T**).

Click on a relationship type to learn more about it.

- [Entailment](https://parlant.io/docs/guidelines/relationships#entailment): When **S** is activated, **T** should always be activated
- [Priority](https://parlant.io/docs/guidelines/relationships#priority): When both **S** and **T** are activated, only **S** should be activated
- [Dependency](https://parlant.io/docs/guidelines/relationships#dependency): When **S** is activated, deactivate it unless **T** is also activated
- [Disambiguation](https://parlant.io/docs/guidelines/relationships#disambiguation): When **S** is activated and two or more of the targets **T ∈ \{T₁, T₂, ...\}** are activated, ask the customer to clarify which action they want to take

### Entailment
> When **S** is activated, **T** should always be activated

```python
await source.entail(target)
```

To understand the need for entailment, we first need to understand how Parlant chooses which guidelines activate for an agent when it's about to say something to the customer.

Basically, Parlant examines the session at its current state, and asks questions about it: "Is this guideline relevant now?", "Is that guideline relevant now?".

To do this, it primarily tests the guidelines' _conditions_.

This would seemingly work well by itself, until you consider two guidelines of the following form:

> * **Guideline A:** When X, Then Y
> * **Guideline B:** When Y, Then Z

Now imagine a situation where, looking at a session, we determine that _X_ does in fact apply, but _Y_ doesn't. With the naive logic above, we would have only fed the agent with the guideline to do _Y_.

But when we step back and analyze this case, we know that the agent is just about to do _Y_, which means that, according to the guidelines we have installed, _Z_ should also apply.

That is what entailment accomplishes: requiring that whenever _A_ is activated, _B_ is also activated.



### Priority
> When both **S** and **T** are activated, only **S** should be activated

```python
await source.prioritize_over(target)
```

Priority can be used for multiple use cases. The two most common ones are:
1. Creating mutually exclusive guidelines
1. Controlling the flow and precedence of actions within the conversation

#### On Controlling Precedence
You may have two guidelines that happen to be activated at the same time, such as:
> * When the customer wants to make a transaction, Then guide them through the process to its completion
> * When the customer has less than $1,000 in their account, Then offer savings plans

You may find that the guidelines above activate simultaneously when, for example, account balance details are introduced into the session while the user is in the process of submitting a transaction.

To ensure that savings plans are offered—but with good timing, only once the transaction is completed—you can prioritize completing the transaction over offering savings plans. Once the transaction is completed, the savings-related guideline may be activated.

### Dependency
> When **S** is activated, deactivate it unless **T** is also activated

```python
await source.depend_on(target)
```

A dependency helps you ensure that a guideline is only activated if other baseline conditions also hold.

The most common use cases is to ensure that more specific conditions are activated only in the proper baseline contexts.

#### Contextualizing Specific Conditions

When you're building flows, you can address specialized or edge-case scenarios by making them dependent on the flow baseline guideline. For example:

##### Baseline Guideline
> When the customer wants to return an order, Then help them complete the return process

##### Dependent Guidelines
> * When the customer isn't able to provide the order number, Then load up their last order's items and ask them to confirm if that is their order
> * When the customer specified the exact order number, Then load up that order's items and ask them to confirm if that is their order

By making these guidelines dependent on the baseline guideline, you can ensure that their evaluation is always performed in the right context.

### Disambiguation
> When **S** is activated and two or more of the targets **T ∈ \{T₁, T₂, ...\}** are activated, ask the customer to clarify which action they want to take

```python
await source.disambiguate([target_1, target_2, ...])
```

You may have a situation where between two (or more) competing guidelines where some or all of which are activated at the same time due to ambiguity, leading to instruction following confusions.

For example, if a customer sent the message _"What are my limits?"_ to a banking agent, and you had the following guidelines, each of which was optimistically activated according to the engine's interpretation:

> * When the customer is inquiring about their ATM limits, Then fetch the data from their account profile
> * When the customer is inquiring about their credit card's limits, Then fetch them from the card provider

To clarify the customer's intent, you could add an [observational guideline](https://parlant.io/docs/guidelines/relationships#observational-guidelines) to disambiguate between the two actions:

```python
ambiguous_limits = await agent.create_observation(
    condition="The customer is inquiring about limits but it isn't clear which kind",
)

await ambiguous_limits.disambiguate([fetch_atm_limits, fetch_credit_card_limits])
```

> **Info: Observations**
>
> `Agent.create_observation()` is a shorthand for creating a guideline without an action. This guideline will still be matched in-context, but it carries no action to perform. It is useful for creating relationships between guidelines in specific scenarios, such as the example above.

## Observational Guidelines
When modeling conversational edge cases, with relationships, you may find yourself wishing to add a guideline just to establish (using its condition) that particular circumstances apply, and—only in those cases—to activate or deactivate other guidelines or journeys using relationships.

To this end, Parlant supports a special type of guideline called an **observational guideline**. This is a guideline that has no action, and is generally only used to establish that certain conditions apply, and to create relationships around them.

```python
observation = await agent.create_observation(condition=CONDITION)
```

You can then use this observation in interesting ways, such as:

1. Deactivating other guidelines by prioritizing the observation over them.
```python
await observation.prioritize_over(other_guideline)
```

2. Scoping other guidelines to only apply when the observation is active.
```python
await other_guideline.depend_on(observation)
```

And other creative uses!
