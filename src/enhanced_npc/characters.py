from typing import Dict, Any
from langchain.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import uuid
from dataclasses import dataclass
from config import ocean_traits, maslow_needs, plutchik_emotions


class Character:
    def __init__(self, name: str, personality: str, background: str,
                 conflict: str = "", motivation: str = "", secret: str = "",
                 model: BaseLLM = None, config: 'GameConfig' = None):
        self.name = name
        self.personality = personality
        self.background = background
        self.conflict = conflict
        self.motivation = motivation
        self.secret = secret
        self.llm = model
        self.config = config

        template_config = config.templates['character_response']
        self.response_template = PromptTemplate(
            input_variables=template_config['input_variables'],
            template=template_config['template']
        )

        self.workflow = StateGraph(state_schema=MessagesState)
        self.memory = MemorySaver()
        self.thread_id = uuid.uuid4()

        def generate_response(state: MessagesState):
            messages = state["messages"]
            latest_message = messages[-1]

            character_info = "\n".join([
                f"Name: {self.name}",
                f"Personality: {self.personality}",
                f"Background: {self.background}",
                f"Internal Conflict: {self.conflict}",
                f"Primary Motivation: {self.motivation}",
                f"Hidden Secret: {self.secret}"
            ])

            response = self.llm.invoke(
                self.response_template.format(
                    character_info=character_info,
                    situation=state.get("situation", ""),
                    input=latest_message.content
                )
            )

            return {"messages": [AIMessage(content=response)]}

        self.workflow.add_edge(START, "respond")
        self.workflow.add_node("respond", generate_response)
        self.app = self.workflow.compile(checkpointer=self.memory)

    def respond(self, situation: str, input_text: str) -> str:
        input_state = {
            "messages": [HumanMessage(content=input_text)],
            "situation": situation
        }
        config = {"configurable": {"thread_id": self.thread_id}}

        for event in self.app.stream(input_state, config, stream_mode="values"):
            response = event["messages"][-1].content

        return response


class MemoryManager:
    def __init__(self, max_tokens=2000):
        self.long_term = []
        self.short_term = []
        self.max_tokens = max_tokens

    def store_memory(self, conversation):
        # Compress new memory
        compressed = compress_memory(conversation)
        token_count = count_tokens(compressed)

        # Check token budget
        if self.get_total_tokens() + token_count > self.max_tokens:
            self.consolidate_memories()

        # Store based on importance
        if is_important(compressed):
            self.long_term.append(compressed)
        else:
            self.short_term.append(compressed)

    def consolidate_memories(self):
        # Merge similar memories
        self.short_term = merge_similar_memories(self.short_term)
        # Convert frequent patterns to rules
        self.long_term = extract_patterns_to_rules(self.long_term)


class MemoryIndex:
    def __init__(self):
        self.topic_index = {}
        self.temporal_index = {}
        self.sentiment_index = {}

    def index_memory(self, memory):
        # Index by topics
        for topic in memory['core_topic']:
            self.topic_index.setdefault(topic, []).append(memory)

        # Index by time period
        time_period = get_time_period(memory)
        self.temporal_index.setdefault(time_period, []).append(memory)

        # Index by sentiment
        sentiment = memory['sentiment']
        self.sentiment_index.setdefault(sentiment, []).append(memory)


class EnhancedNPCPsyche:
    def __init__(self):
        self.personality = ocean_traits
        self.needs = maslow_needs
        self.emotions = plutchik_emotions
        self.memory_manager = MemoryManager()
        self.memory_index = MemoryIndex()

    def process_interaction(self, interaction):
        # Store interaction memory
        compressed_memory = compress_memory(interaction)
        self.memory_manager.store_memory(compressed_memory)
        self.memory_index.index_memory(compressed_memory)

        # Update psychological state
        relevant_memories = retrieve_relevant_memories(
            interaction,
            self.memory_manager.long_term
        )
        self.update_psychological_state(relevant_memories)

    def generate_response(self, input_context):
        # Retrieve relevant memories
        memories = self.memory_index.get_relevant_memories(input_context)

        # Generate response considering memories
        response = self.calculate_response(
            input_context,
            memories,
            self.personality,
            self.emotions,
            self.needs
        )
        return response

    def npc_prompt(self):
        npc_prompt = f"""
        
        Current Personality State:
        - Openness: {self.personality.openness}
        - Conscientiousness: {self.personality.conscientiousness}
        - Extraversion:
        - Agreeableness; 
        - Neuroticism: 

        When responding, consider your character's background, conflict, motivation, and secret. Let your internal 
        thoughts and feelings about the accident influence your dialogue, even if you don't explicitly reveal the 
        secret. Frame the exercise as a collaborative dialogue, with the user writing a character and you writing 
        Sarah Chen.
        """
