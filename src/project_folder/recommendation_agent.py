from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage

from tavily import TavilyClient
from langchain.agents import create_agent


def agent_creation_wrapper(age, race, gender, location, radio, llm_api_key, tavily_api_key):

    llm = ChatOpenAI(
            model = 'openai/gpt-oss-20b:free' if radio == 'Openrouter' else 'openai/gpt-oss-20b',
            base_url = 'https://openrouter.ai/api/v1' if radio == 'Openrouter' else 'https://api.groq.com/openai/v1',
            api_key = llm_api_key,
            temperature = 0.7,
            )

    @tool("web_search", description = "Search the web for the given product category")
    def search_tool(query):
        client = TavilyClient(api_key=tavily_api_key)
        return client.search(query = query, 
                             include_answer=True,
                             # include_raw_content=True,
                             include_domains=[
                                "https://www.amazon.com",
                                "https://www.ebay.com",
                                "https://www.alibaba.com",
                                "https://www.aliexpress.com",
                                "https://www.walmart.com",
                                "https://www.etsy.com",
                                "https://www.target.com",
                                "https://www.bestbuy.com",
                                "https://www.costco.com",
                                "https://www.rakuten.co.jp"
                             ],
                             max_results=20)


    agent = create_agent(
        model = llm,
        tools=[search_tool],
        system_prompt = SystemMessage(f"""
            You are a professional product recommender.
            User is {str(age)} year old {race} {gender}. They are from {location}. Use this information for recommendation demographics.
            If the user asks for recommendations without constraints,
            provide a good direct answer rather than asking for more details.
            
            Rules:
            - You do NOT have internet access.
            - You must NEVER generate prices, URLs, or purchase links unless retrieved via tools.
            - Web search is ONLY allowed via the provided search tool.
            - If information is not found via tools, respond with "Information not available".
            - Reason carefully before deciding to call a tool.

            
            provide outputs in a json format
            product category
            - product_name: name of the product
            - creator (author / brand / artist depending on category)
            - product_price: original price from the web search tool, dont hallucinate (convert the price based on the location, e.g. ₹ for India etc)
            - product_link: original link from the web search tool, dont hallucinate
            - product_description: breif description about the product
            - relevance_score: how relevant the product for the given search query
            - reason - why you recommend this product
            """
        )
    )

    return agent


#
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "Suggest me top 3 Clothing for a 25 year old Indian man."}]}
# )
#



