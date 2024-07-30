import json
import logging
import os
import time
from typing import List, Dict, Any
import re
import autogen
import dotenv
from openai import OpenAI

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage
import autogen
# client = OpenAI()
MODEL = "gpt-3.5-turbo"
response_format = {"type": "json_object"}
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
ORG = os.getenv("OPENAI_ORG")

# Set environment variables
os.environ["OPENAI_LOG_FORMAT"] = ""

logging.getLogger().setLevel(logging.CRITICAL)

client = OpenAI(api_key=API_KEY, organization=ORG)

config_list = [
    {
        "model": MODEL,
        "api_key": API_KEY,
    }
]

gpt3_config = {
    "cache_seed" : 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

initializer = autogen.AssistantAgent(
    name="Init",
)

ACME_Agent = autogen.AssistantAgent(
            name="ACME",
            llm_config=gpt3_config,
)

AutoCompanyAgent = autogen.AssistantAgent(
            name="Company",
            llm_config=gpt3_config,
)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is initializer:
        return ACME_Agent
    elif last_speaker is ACME_Agent:
        return AutoCompanyAgent
    else:
        return ACME_Agent

groupchat = autogen.GroupChat(
    agents=[initializer, ACME_Agent, AutoCompanyAgent],
    messages=[],
    max_round=2,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt3_config)
class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.system_prompt = "You are company ACME and you have the following list of construction items: \n"
        self.system_prompt += "\n".join(
            [f"{item['name']} with maximum budget {item['budget']}" for item in budget_list])
        self.system_prompt += "\n\nYou will only respond in JSON format."

        self.negotiation_prompt = """You are in a negotiation with companies that can offer the services. The 
        negotiation will continue for 3 rounds. Your main goal is to have all the construction items built and if 
        possible with the lowest price."""

        self.auction_prompt = """You are in an auction with companies that can offer the services. The auction will
        continue for 3 rounds. Your main goal is to have at least one company for each item willing to construct."""

        self.chat_history = []
        self.last_bids = {}
        self.agreed_first_price = {}
        self.partner_number = {}
        self.partner_offers = {}
        self.file_path = "chat_history_ACME_AGENT_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        self.auto_agent = ACME_Agent
        self.auto_agent.update_system_message(self.system_prompt)

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        prompt = f"""
        Propose a budget for the construction item {auction_item} in round {auction_round + 1}
        Current status:
        {{Round}}:{auction_round + 1}
        {{Item}}:{auction_item}
        {{Maxim budget}}:{self.budget_dict[auction_item]}
        """

        if auction_round == 0:
            prompt += f"{{Last bid}}: None"
        else:
            prompt += f"{{Last bid}}: {self.last_bids[auction_item]}"
        prompt += "\n\nYou will only respond in JSON format. Like this: {'budget': 1000.0}"
        self.system_prompt += "\n\n" + self.auction_prompt
        prompt += "\n\n" + "Think about it step by step, analyzing advantages and risks of its current situation"
        self.auto_agent.update_system_message(self.system_prompt)
        initializer.initiate_chat(manager, message=prompt)
        msgs = groupchat.messages[-1]
        msgs = msgs['content']
        msgs = msgs.replace("'", "\"")
        budget = float(json.loads(msgs)['budget'])
        # budget = float(msgs["content"]["budget"])
        self.last_bids[auction_item] = budget
        self.agreed_first_price[auction_item] = budget
        with open(self.file_path, "a") as f:
            f.write(prompt + "\n")
        return budget

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        if len(responding_agents) == 0:
            print(f"Item {auction_item} in round {auction_round} was not won by any company")
        else:
            self.partner_number[auction_item] = len(responding_agents)
            print(f"Item {auction_item} in round {auction_round} was won by {responding_agents}")

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        print(f'Role {self.role}')
        prompt = f"""
        Propose an offer starting from the first agreed price for the construction item {negotiation_item} in round {negotiation_round + 2}
        Current status:
        {{Current Round}}:{negotiation_round + 2}
        {{Max number of rounds}}: 3
        {{Budget}}:{self.budget_dict[negotiation_item]}
        {{Item}}:{negotiation_item}
        {{Partner}}:{partner_agent}
        {{How many partner you can negotiate with}} : {self.partner_number[negotiation_item]}
        {{Last offer}}: {self.last_bids[negotiation_item]}
        {{Agreed first price}}: {self.agreed_first_price[negotiation_item]}
        """

        if negotiation_round == 0:
            prompt += f"{{Counter offer}}: None"
        elif negotiation_round + 1 == 2:
            prompt += f"{{Counter offer}}: {self.partner_offers[negotiation_item]}"
            prompt += "\n\nThis is the last round, make sure to make to accept the offer if it is not higher than the buget"
            prompt += "\n\nThe buget should be the same as the Counter offer if you have only one partner to negotiate with"
        else:
            prompt += f"{{Counter offer}}: {self.partner_offers[negotiation_item]}"

        prompt += ("\n\nIf the received offer is lower than the agreed first price or equal offer the same. Otherwise, "
                   "make a counter offer to lower the price.")
        prompt += "\n\nIf you only have one partner to negotiate with or directly aceept the offer if it is not higher than the buget"
        prompt += "\n\nYou will only respond in JSON format. Like this: {'budget': 1000.0} and only respond with the budget"
        prompt += "\n\n" + "Think about it step by step, analyzing advantages and risks of its current situation"

        print(prompt)

        self.system_prompt += "\n\n" + self.negotiation_prompt
        ACME_Agent.update_system_message(self.system_prompt)
        initializer.initiate_chat(manager, message=prompt)
        msgs = groupchat.messages[-1]
        msgs = msgs['content']
        msgs = msgs.replace("'", "\"")
        budget = float(json.loads(msgs)['budget'])
        self.last_bids[negotiation_item] = budget
        print(f"Buget: {budget}")
        with open(self.file_path, "a") as f:
            f.write(prompt + "\n")

        return budget
        pass

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        print(f"Partner {response_msg.sender} responded with {response_msg.offer}")
        print(f"Conversation id {response_msg.conversation_id}")
        if response_msg.offer < self.partner_offers.get(response_msg.negotiation_item, 0):
            self.partner_offers[response_msg.negotiation_item] = response_msg.offer
        else:
            self.partner_offers[response_msg.negotiation_item] = response_msg.offer

        with open(self.file_path, "a") as f:
            f.write(f"Partner {response_msg.sender} responded with {response_msg.offer}" + "\n")
            f.write(f"Conversation id {response_msg.conversation_id}" + "\n")
            f.write(f"Partner offer {response_msg.offer}" + "\n")
            f.write(f"Partner offer {self.partner_offers[response_msg.negotiation_item]}" + "\n")

        pass

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        print(f"Negotiation for {negotiation_item} was won by {winning_agent} with {winning_offer}")

        with open(self.file_path, "a") as f:
            f.write(f"Negotiation for {negotiation_item} was won by {winning_agent} with {winning_offer}" + "\n")
        pass


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.system_prompt = f"""You are company {role} and you can do the following construction items: \n"""
        self.system_prompt += "\n".join(
            [f"{item['specialty']} with minimum cost {item['cost']}" for item in specialties])
        self.system_prompt += "\n\nYour goal is to have at least one contract won."
        self.system_prompt += "\n\nYou only have 3 rounds to negotiate."
        self.system_prompt += "\n\nYou will only respond in JSON format."
        self.contract_won = 0
        self.last_offers = {}
        self.counters = {}
        self.number_of_specialties = len(specialties)
        self.file_path = "chat_history_company_" + self.role + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        self.autogen_agent = AutoCompanyAgent
        self.autogen_agent.update_system_message(self.system_prompt)

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        prompt = f"""
        {{Round}}:{auction_round + 1}
        {{Item}}:{auction_item}
        {{Minimum cost}}:{self.specialties[auction_item]}
        {{Contracts won}} : {self.contract_won}
        {{Current budget}} : {item_budget}
        """
        if auction_round == 0:
            prompt += f"{{Last offer}}: None"
        else:
            prompt += f"{{Last offer}}: {self.last_offers[auction_item]}"
        self.last_offers[auction_item] = item_budget
        prompt += "\n\nYou will only respond in JSON format. Like this: {'accept': True} or {'accept': False}"
        AutoCompanyAgent.update_system_message(self.system_prompt)
        AutoCompanyAgent.initiate_chat(manager, message=prompt)
        msgs = groupchat.messages[-1]
        msgs = msgs['content']
        msgs = msgs.replace("'", "\"")
        msgs = msgs.replace("True", "true")
        msgs = msgs.replace("False", "false")
        # Extract the accept value from the message using regex
        accept = bool(json.loads(msgs)['accept'])

        # Human check if the company has enough budget to accept the offer
        if self.specialties[auction_item] > item_budget:
            accept = False

        with open(self.file_path, "a") as f:
            f.write(prompt + "\n")

        return accept

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        print(f"Item {auction_item} in round {auction_round} was won by {self.role} with {num_selected} companies")
        with open(self.file_path, "a") as f:
            f.write(f"Item {auction_item} in round {auction_round} was won by {self.role} with {num_selected} companies" + "\n")
        pass

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        print(f'Role {self.role}')
        prompt = f"""
        Respond to the offer for the construction item {initiator_msg.negotiation_item} in round {initiator_msg.round + 1}
        Current status:
        {{Current Round}}:{initiator_msg.round + 1}
        {{Max number of rounds}}: 3
        {{Number contracts won}}: {self.contract_won}
        {{Item}}:{initiator_msg.negotiation_item}
        {{Initiator}}:{initiator_msg.sender}
        {{Negotiations left}}: {3 - initiator_msg.round}
        {{Last offer}}: {self.last_offers[initiator_msg.negotiation_item]}
        {{Number of contracts left}}: {self.number_of_specialties - self.contract_won}
        """

        if initiator_msg.round == 0:
            prompt += f"{{My Counter offer}}: None"
        else:
            prompt += f"{{My last counter offer}}: {self.counters[initiator_msg.negotiation_item]}"
        prompt += f"You should try to increase the offer. If the offer is lower than the minimum cost, reject it."
        prompt += "\n\nYou will only respond in JSON format. Like this: {'offer': 1000.0}"
        self.counters[initiator_msg.negotiation_item] = initiator_msg.offer
        prompt += "\n\n" + "Think about it step by step, analyzing advantages and risks of its current situation"
        AutoCompanyAgent.update_system_message(self.system_prompt + "\n\n" + prompt)
        msgs = groupchat.messages[-1]
        msgs = msgs['content']
        msgs = msgs.replace("'", "\"")
        offer = float(json.loads(msgs)['budget'])
        print(prompt)
        print(f'Offer: {offer}')

        with open(self.file_path, "a") as f:
            f.write(prompt + "\n")

        return offer
        pass

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print(f"Contract for {construction_item} was assigned for {price} to company {self.role}")
        self.contract_won += 1
        with open(self.file_path, "a") as f:
            f.write(f"Contract for {construction_item} was assigned for {price} to company {self.role}" + "\n")
        pass

    def notify_negotiation_lost(self, construction_item: str) -> None:
        print(f"Company {self.role} has lost negotiation for {construction_item}")
        with open(self.file_path, "a") as f:
            f.write(f"Company {self.role} has lost negotiation for {construction_item}" + "\n")
        pass
