# Mutli-Agent Construction Auction and Negotiation using Prompt Engineering

## Overview
This project focuses on developing agents for ACME and construction companies to facilitate auctions and negotiations for construction items using prompt engineering. The agents use various prompts to propose budgets, offer bids, and negotiate contracts. The AutoGen framework is used to manage interactions between these agents.

## Table of Contents
1. [Prompt Engineering](#prompt-engineering)
   - [ACME Agent](#acme-agent)
   - [Company Agent](#company-agent)
2. [Justifications](#justifications)
   - [ACME Agent](#acme-agent-justification)
   - [Company Agents](#company-agents-justification)
3. [AutoGen Implementation](#autogen-implementation)
   - [Agents and Interaction](#agents-and-interaction)
   - [Single Company Interaction](#single-company-interaction)
   - [Double Company Interaction](#double-company-interaction)

## Prompt Engineering

### ACME Agent
The ACME agent is designed with a main system prompt and specific prompts for auction and negotiation phases.

#### System Prompt
The system prompt initializes with a list of construction items and their maximum budgets, and specifies that responses should be in JSON format.

#### Auction Prompt
In the auction phase, the ACME agent aims to secure at least one company for each construction item over three rounds.

#### Negotiation Prompt
In the negotiation phase, the goal is to have all construction items built at the lowest possible price over three rounds.

### Company Agent
The company agent also has a main system prompt and additional prompts for bidding and responding to offers.

#### System Prompt
The system prompt lists the company's specialties and minimum costs, with the goal of winning at least one contract within three rounds.

#### Bidding Prompt
During bidding, the company agent decides whether to accept an offer based on its budget, minimum costs, and contracts won.

#### Offer Prompt
When responding to an offer, the company agent considers the initiator's details, contracts won, and negotiations left, and aims to negotiate the best possible price.

## Justifications

### ACME Agent Justification
- **Propose a budget:** The ACME agent proposes a budget considering the need to stay competitive while covering essential costs. For instance, a budget of 3000 is proposed against a maximum budget of 5000 to balance cost-saving and risk.

### Company Agents Justification
- **Accept or Reject Offers:** Company agents decide based on their minimum costs and current budgets. For example, if a budget is below the minimum cost, the offer is rejected.

## AutoGen Implementation

### Agents and Interaction
Three main agents are created using the AutoGen framework:
- **Initializer**
- **ACME Agent**
- **Company Agent**

A manager agent handles the central communication and state transitions between agents.

#### State Transition Function
```python
def state_transition(last_speaker, groupchat):
    if last_speaker is initializer:
        return ACME_Agent
    elif last_speaker is ACME_Agent:
        return AutoCompanyAgent
    elif last_speaker is AutoCompanyAgent:
        return AutoCompanyAgent
    else:
        return ACME_Agent
```

#### Agent Configuration
```python
initializer = autogen.AssistantAgent(name="Init")
ACME_Agent = autogen.AssistantAgent(name="ACME", llm_config=gpt3_config)
AutoCompanyAgent = autogen.AssistantAgent(name="Company", llm_config=gpt3_config)
```
#### Single Company Interaction

In a single company scenario, the negotiation process spans three rounds, with the company aiming to win the contract at the best price.
```
Auction stage:
- Round 1: ACME proposes a budget of 3000.
- Company evaluates and either accepts or rejects based on minimum cost and budget.

Negotiation stage:
- ACME and Company negotiate the final contract price, aiming for mutual agreement within budget constraints.
```

#### Double Company Interaction
When two companies are involved, negotiations are faster, and strategies differ based on contract wins and profit maximization.

```
Auction stage:
- Round 1: ACME proposes a budget of 3000.
- Companies evaluate and respond based on their minimum costs and current budgets.

Negotiation stage:
- ACME negotiates with both companies, considering offers and counteroffers to secure the best deals.
```

## Conclusion
This project demonstrates the application of prompt engineering and the AutoGen framework to simulate realistic auction and negotiation scenarios. The agents' interactions and decision-making processes are designed to achieve optimal outcomes for both ACME and the construction companies.