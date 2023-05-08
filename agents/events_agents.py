

import pandas as pd
from langchain import OpenAI, PromptTemplate
from langchain.agents import (create_pandas_dataframe_agent)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.utilities import WikipediaAPIWrapper


HUMAN_MESSAGE_PROMPT = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""
                Construct an 
                Please parse the Wikipedia text and extract events from it in the following format. Please return only the JSON format. DONT use any additional text except JSON
                The JSON object contains the following fields: title, start_date, end_date, description
                The wiki text is following: {context}?""",
            input_variables=["context"],
        )
    )

def construct_search_chain(user_input: str) -> str:
    return f"""
        Construct a reasoning chain for this multi-hop question [Question]: {user_input}
        You should generate a query to the search engine based on what you already know at each step of the reasoning chain, starting with [Query].
        If you know the answer for [Query], generate it starting with [Answer].
        You can try to generate the final answer for the [Question] by referring the [Query]-[Answer] pairs, starting with [Final Content].
        If you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with [Unsolved Query].
        For example:
        [Question]: {user_input}
    """

def run(user_input: str, df: pd.DataFrame) -> str:
    chat_prompt_template = ChatPromptTemplate.from_messages([HUMAN_MESSAGE_PROMPT])
    llm = ChatOpenAI(temperature=0)
    event_chain = LLMChain(llm=llm, prompt=chat_prompt_template)

    pandas_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    wiki_agent = WikipediaAPIWrapper()

    wiki_response = wiki_agent.run(user_input)
    event = event_chain.run(wiki_response)

    return pandas_agent.run(f"{user_input}. Take date range from {event}")