import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

import sys
sys.path.append('.')
from envs.urban_config import *

os.environ["OPENAI_API_KEY"] = ''

class ActionAgent:
    def __init__(
            self,
            model: ChatOpenAI,
            system_message: SystemMessage,
    ) -> None:
        self.model = model
        self.system_message = system_message

    def send(self, human_message) -> str:
        prompt = [self.system_message, human_message]
        # print(prompt)
        message = self.model(prompt).content  #
        return message


def create_LLM_agent(model, prompt):
    system_message = SystemMessage(
        content="\n\n".join(["MISSION:\n" + prompt["mission"]]
                            + ["RULES:\n" + prompt["rules"]]
                            + ["PLANNING GOALS:\n" + prompt["planning goals"]]
                            + ["INSTRUCTIONS:\n" + prompt["instructions"]]
                            + ["PLANNING TARGETS"])
    )
    agent = ActionAgent(model, system_message)
    return agent


def LLM_select_action(area_type, live_circle_info_list, global_info_dict, LLM_agent) -> list:
    """
    Use live_circle_info_list and global_info_dict to generate prompts and sort ActionAgent to select actions
    :param live_circle_info_list:
    :param global_info_dict:
    :return: action_candidate
    """
    human_message = HumanMessage(
        content="There is a " + area_type + " area. Surrounding functional types information within the living circle: " + str(live_circle_info_list) + "\n"
                + "Current coverage area ratio of each functional types in the community: "
                + "Business area is " + str(global_info_dict['Business']) + ', '
                + "Recreation area is " + str(global_info_dict['Recreation']) + ', '
                + "Office area is " + str(global_info_dict['Office']) + ', '
                + "Park area is " + str(global_info_dict['Park']) + ', '
                + "Openspace area is " + str(global_info_dict['OpenSpace']) + '. '
                + "Current number of schools and hospitals in the community: "
                + "Small school(s) is(are) " + str(global_info_dict['small_school_num']) + ', '
                + "Large school(s) is(are) " + str(global_info_dict['large_school_num']) + ', '
                + "Small hospital(s) is(are) " + str(global_info_dict['small_hospital_num']) + ', '
                + "Large hospital(s) is(are) " + str(global_info_dict['large_hospital_num']) + '. '
                + "\n\n" + "What's your suggestion?"
    )
    answer = LLM_agent.send(human_message)
    print(human_message)
    print(answer)

    if answer[0] == "[":
        try:
            answer = answer.split('[')[1].split(']')[0].split(', ')
            action_candidate = []
            new_func_types = {v: k for k, v in FUNC_TYPES.items()}
            for types in range(len(answer)):
                if answer[types][0] == "'":
                    action_candidate.append(int(new_func_types[answer[types][1:-1]]))
                else:
                    action_candidate.append(int(new_func_types[answer[types]]))
        except:
            action_candidate = [i for i in range(1, len(FUNC_TYPES))]
    else:
        action_candidate = [i for i in range(1, len(FUNC_TYPES))]
    action_candidate = list(set(action_candidate))
    return action_candidate

