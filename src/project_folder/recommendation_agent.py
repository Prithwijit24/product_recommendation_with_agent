from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from ddgs import DDGS
import json
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List

logger = logging.getLogger(__name__)


# Pydantic Models for structured output
class Product(BaseModel):
    """Model for a single product recommendation"""
    product_name: str = Field(..., description="Exact name of the product")
    creator: str = Field(..., description="Brand, author, or manufacturer")
    product_price: str = Field(..., description="Price with currency symbol or 'Not available'")
    product_link: str = Field(..., description="Full URL from search results")
    product_description: str = Field(..., description="1-2 sentence description")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score between 0 and 1")
    reason: str = Field(..., description="Why this product is recommended")
    
    @field_validator('relevance_score')
    @classmethod
    def validate_score(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('relevance_score must be between 0 and 1')
        return v


class RecommendationResponse(BaseModel):
    """Model for the complete recommendation response"""
    product_category: str = Field(..., description="Category of recommended products")
    recommended_products: List[Product] = Field(..., description="List of recommended products")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "product_category": "Clothing",
                "recommended_products": [
                    {
                        "product_name": "Men's Cotton Shirt",
                        "creator": "BrandX",
                        "product_price": "₹500",
                        "product_link": "https://example.com/shirt",
                        "product_description": "High-quality cotton shirt for everyday wear",
                        "relevance_score": 0.95,
                        "reason": "Perfect for your age group and style preferences"
                    }
                ]
            }
        }
    }


def agent_creation_wrapper(age, race, gender, location, radio, llm_api_key, tavily_api_key=None):
    """
    Create a product recommendation agent with DuckDuckGo search and Pydantic validation.
    
    Parameters:
    - age, race, gender, location: User demographic information
    - radio: 'Openrouter' or 'Grok' (determines LLM endpoint)
    - llm_api_key: API key for the LLM
    - tavily_api_key: Deprecated (kept for backward compatibility)
    
    Returns:
    - agent_executor: Configured agent executor with structured output
    """

    llm = ChatOpenAI(
        model='openrouter/auto' if radio == 'Openrouter' else 'mixtral-8x7b-32768',
        base_url='https://openrouter.ai/api/v1' if radio == 'Openrouter' else 'https://api.groq.com/openai/v1',
        api_key=llm_api_key,
        temperature=0.7,
    )

    @tool("product_search", description="Search for products across e-commerce platforms. Returns product names, links, prices, and descriptions.")
    def search_tool(query: str) -> str:
        """
        Search for products using DuckDuckGo.
        Returns structured product information.
        """
        try:
            ddgs = DDGS()
            results = ddgs.text(
                keywords=query,
                region='wt-wt',
                safesearch='moderate',
                timelimit='w',
                max_results=15
            )
            
            if not results:
                return json.dumps({"error": "No products found", "query": query})
            
            # Format results for the LLM
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", "N/A"),
                    "url": result.get("href", "N/A"),
                    "snippet": result.get("body", "N/A")[:200]  # Limit snippet length
                })
            
            return json.dumps(formatted_results)
        
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return json.dumps({"error": f"Search failed: {str(e)}"})

    # Create the system message for the agent
    system_message = f"""You are a professional product recommender specializing in personalized recommendations.

USER PROFILE:
- Age: {age} years old
- Gender: {gender}
- Ethnicity/Race: {race}
- Location: {location}

CRITICAL INSTRUCTIONS:
1. DO NOT ask any questions. Provide recommendations directly based on the request and user profile.
2. Use the product_search tool to find relevant products for the requested category and budget.
3. Search for products that match the user's age, gender, location, and preferences.
4. Always retrieve actual product information via the search tool - NEVER hallucinate prices, URLs, or product details.
5. Provide exactly 3-5 product recommendations per request.
6. Ensure all relevance_score values are between 0 and 1.

MANDATORY OUTPUT FORMAT - Return ONLY this exact JSON structure, no additional text:
{{
    "product_category": "category name",
    "recommended_products": [
        {{
            "product_name": "exact product name",
            "creator": "brand/author/manufacturer",
            "product_price": "price with currency symbol (e.g., ₹500, $50) or 'Not available'",
            "product_link": "full URL from search results",
            "product_description": "1-2 sentence description",
            "relevance_score": 0.9,
            "reason": "why this product matches the user profile"
        }}
    ]
}}

BEHAVIOR RULES:
- Always provide recommendations immediately without asking for clarification
- Match location and currency (e.g., rupees for India, dollars for USA)
- Only include information found via search. Never hallucinate prices, links, or specifications
- If product price is not available, use "Price not available"
- relevance_score MUST be a decimal number between 0 and 1 (e.g., 0.85, not 85)
- For authors/creators in book recommendations, always include author names
- Be concise but informative in descriptions (1-2 sentences)
- Return ONLY the JSON, no markdown code blocks or additional text"""

    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_message
    )

    return agent


def parse_and_validate_response(response_text: str) -> RecommendationResponse:
    """
    Parse and validate agent response using Pydantic.
    
    Parameters:
    - response_text: Raw response text from the agent
    
    Returns:
    - RecommendationResponse: Validated recommendation response
    
    Raises:
    - ValueError: If response cannot be parsed or validated
    """
    try:
        # Clean up response (remove markdown code blocks if present)
        json_str = response_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Validate using Pydantic model
        recommendation = RecommendationResponse(**data)
        
        logger.info(f"Successfully validated recommendation response with {len(recommendation.recommended_products)} products")
        return recommendation
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        raise ValueError(f"Invalid JSON in response: {str(e)}")
    except ValueError as e:
        logger.error(f"Pydantic validation error: {str(e)}")
        raise ValueError(f"Response validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {str(e)}")
        raise ValueError(f"Failed to parse response: {str(e)}")
#
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "Suggest me top 3 Clothing for a 25 year old Indian man."}]}
# )
#



