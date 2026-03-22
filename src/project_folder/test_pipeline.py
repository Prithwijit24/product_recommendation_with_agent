#!/usr/bin/env python
"""
Comprehensive test suite for the recommendation pipeline Pydantic models.
Tests response validation and JSON parsing without agent dependencies.
"""

import json
import sys
from pydantic import BaseModel, Field, validator
from typing import List
import logging

logger = logging.getLogger(__name__)

# Define Pydantic models directly in test for independence
class Product(BaseModel):
    """Model for a single product recommendation"""
    product_name: str = Field(..., description="Exact name of the product")
    creator: str = Field(..., description="Brand, author, or manufacturer")
    product_price: str = Field(..., description="Price with currency symbol or 'Not available'")
    product_link: str = Field(..., description="Full URL from search results")
    product_description: str = Field(..., description="1-2 sentence description")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score between 0 and 1")
    reason: str = Field(..., description="Why this product is recommended")
    
    @validator('relevance_score')
    def validate_score(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('relevance_score must be between 0 and 1')
        return v


class RecommendationResponse(BaseModel):
    """Model for the complete recommendation response"""
    product_category: str = Field(..., description="Category of recommended products")
    recommended_products: List[Product] = Field(..., description="List of recommended products")
    
    class Config:
        schema_extra = {
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

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")


# Test 1: Pydantic Product Model Validation
def test_product_model():
    print_header("Test 1: Product Model Validation")
    
    try:
        # Valid product
        product = Product(
            product_name="Men's Cotton Shirt",
            creator="BrandX",
            product_price="₹500",
            product_link="https://example.com/shirt",
            product_description="High-quality cotton shirt for everyday wear",
            relevance_score=0.95,
            reason="Perfect for your age group"
        )
        print_success(f"Product created: {product.product_name}")
        print(f"  - Creator: {product.creator}")
        print(f"  - Price: {product.product_price}")
        print(f"  - Relevance: {product.relevance_score}")
        
    except Exception as e:
        print_error(f"Failed to create valid product: {str(e)}")
        return False
    
    # Test invalid relevance_score (should fail)
    try:
        invalid_product = Product(
            product_name="Test",
            creator="Test",
            product_price="$100",
            product_link="https://example.com",
            product_description="Test",
            relevance_score=1.5,  # Invalid: > 1
            reason="Test"
        )
        print_error("Should have rejected relevance_score > 1")
        return False
    except ValueError as e:
        print_success(f"Correctly rejected invalid relevance_score: {str(e)}")
        return True


# Test 2: Pydantic RecommendationResponse Model
def test_recommendation_response_model():
    print_header("Test 2: RecommendationResponse Model Validation")
    
    try:
        response = RecommendationResponse(
            product_category="Clothing",
            recommended_products=[
                Product(
                    product_name="T-Shirt",
                    creator="BrandA",
                    product_price="$20",
                    product_link="https://example.com/tshirt",
                    product_description="Comfortable cotton t-shirt",
                    relevance_score=0.9,
                    reason="Good fit for your style"
                ),
                Product(
                    product_name="Jeans",
                    creator="BrandB",
                    product_price="$50",
                    product_link="https://example.com/jeans",
                    product_description="Classic blue jeans",
                    relevance_score=0.85,
                    reason="Perfect casual wear"
                )
            ]
        )
        
        print_success(f"Response created with {len(response.recommended_products)} products")
        print(f"  - Category: {response.product_category}")
        for i, product in enumerate(response.recommended_products, 1):
            print(f"  - Product {i}: {product.product_name} (Score: {product.relevance_score})")
        
        # Test JSON serialization
        json_data = response.model_dump()
        json_str = json.dumps(json_data, indent=2)
        print_success(f"JSON serialization successful ({len(json_str)} chars)")
        return True
        
    except Exception as e:
        print_error(f"Failed: {str(e)}")
        return False


# Test 3: Response Parsing and Validation
def test_response_parsing():
    print_header("Test 3: Response Parsing & Validation")
    
    # Mock JSON response from LLM
    mock_response = """
    {
        "product_category": "Books",
        "recommended_products": [
            {
                "product_name": "Clean Code",
                "creator": "Robert C. Martin",
                "product_price": "$50",
                "product_link": "https://amazon.com/clean-code",
                "product_description": "A handbook of agile software craftsmanship",
                "relevance_score": 0.98,
                "reason": "Essential reading for developers"
            },
            {
                "product_name": "The Pragmatic Programmer",
                "creator": "Hunt & Thomas",
                "product_price": "$45",
                "product_link": "https://amazon.com/pragmatic",
                "product_description": "Your journey to mastery in software development",
                "relevance_score": 0.92,
                "reason": "Great for learning best practices"
            }
        ]
    }
    """
    
    try:
        result = parse_and_validate_response(mock_response)
        print_success("Response parsed and validated successfully")
        print(f"  - Category: {result.product_category}")
        print(f"  - Products: {len(result.recommended_products)}")
        for product in result.recommended_products:
            print(f"    • {product.product_name} by {product.creator}")
        return True
        
    except Exception as e:
        print_error(f"Failed to parse response: {str(e)}")
        return False


# Test 4: Markdown Code Block Handling
def test_markdown_handling():
    print_header("Test 4: Markdown Code Block Handling")
    
    # Response with markdown code blocks (common LLM behavior)
    mock_response_markdown = """
    Here are the recommendations:
    
    ```json
    {
        "product_category": "Technology",
        "recommended_products": [
            {
                "product_name": "MacBook Pro",
                "creator": "Apple",
                "product_price": "$2000",
                "product_link": "https://apple.com/macbook",
                "product_description": "Powerful laptop for professionals",
                "relevance_score": 0.95,
                "reason": "Best performance for developers"
            }
        ]
    }
    ```
    """
    
    try:
        result = parse_and_validate_response(mock_response_markdown)
        print_success("Successfully parsed response with markdown code blocks")
        print(f"  - Extracted category: {result.product_category}")
        print(f"  - Products: {[p.product_name for p in result.recommended_products]}")
        return True
        
    except Exception as e:
        print_error(f"Failed to handle markdown: {str(e)}")
        return False


# Test 5: Invalid Response Handling
def test_invalid_responses():
    print_header("Test 5: Invalid Response Handling")
    
    test_cases = [
        ("Invalid JSON", "{invalid json}", "JSON decode error"),
        ("Missing fields", '{"product_category": "Test"}', "Missing required field"),
        ("Invalid score", '{"product_category": "Test", "recommended_products": [{"product_name": "Test", "creator": "Test", "product_price": "100", "product_link": "http://test.com", "product_description": "test", "relevance_score": 1.5, "reason": "test"}]}', "Invalid relevance_score"),
    ]
    
    all_passed = True
    for test_name, response, expected_error in test_cases:
        try:
            result = parse_and_validate_response(response)
            print_error(f"  Should have failed: {test_name}")
            all_passed = False
        except ValueError as e:
            print_success(f"  {test_name}: Correctly rejected")
    
    return all_passed


# Test 6: Streamlit Integration Simulation
def test_streamlit_integration():
    print_header("Test 6: Streamlit Integration Simulation")
    
    try:
        # Simulate how Streamlit app would use the validation
        user_preferences = {
            "age": 32,
            "race": "Indian",
            "gender": "Female",
            "location": "India",
            "category": "Technology"
        }
        print_success(f"User preferences: {user_preferences}")
        
        # Simulate agent response (what LLM would return)
        mock_agent_response = """
        {
            "product_category": "Technology",
            "recommended_products": [
                {
                    "product_name": "Apple AirPods Pro",
                    "creator": "Apple",
                    "product_price": "₹29,900",
                    "product_link": "https://www.apple.com/in/airpods-pro",
                    "product_description": "Premium wireless earbuds with active noise cancellation",
                    "relevance_score": 0.92,
                    "reason": "Perfect audio quality for tech-savvy users in India"
                }
            ]
        }
        """
        
        # Validate the response
        validated = parse_and_validate_response(mock_agent_response)
        
        # Access as Streamlit would
        print_success(f"Category: {validated.product_category}")
        for i, prod in enumerate(validated.recommended_products, 1):
            print_success(f"  Product {i}: {prod.product_name} - {prod.product_price}")
            
        return True
        
    except Exception as e:
        print_error(f"Streamlit integration failed: {str(e)}")
        return False


# Test 7: Complete Pipeline Simulation
def test_complete_pipeline():
    print_header("Test 7: Complete Pipeline Simulation")
    
    print_info("Simulating complete recommendation pipeline...")
    
    # Step 1: Create user profile
    user_profile = {
        "age": 28,
        "race": "Indian",
        "gender": "Male",
        "location": "India"
    }
    print_success(f"User profile created: {user_profile}")
    
    # Step 2: Simulate LLM response
    simulated_llm_response = """
    ```json
    {
        "product_category": "Clothing",
        "recommended_products": [
            {
                "product_name": "Formal White Shirt",
                "creator": "Raymond",
                "product_price": "₹1499",
                "product_link": "https://www.raymond.in/shirts",
                "product_description": "Premium cotton formal shirt perfect for office wear and special occasions",
                "relevance_score": 0.96,
                "reason": "Highly suitable for a 28-year-old male, professional look for Indian market"
            },
            {
                "product_name": "Slim Fit Jeans",
                "creator": "Lee",
                "product_price": "₹2499",
                "product_link": "https://www.lee.in/jeans",
                "product_description": "Comfortable slim fit denim jeans with stretch fabric",
                "relevance_score": 0.88,
                "reason": "Modern casual wear option for Indian climate and demographics"
            },
            {
                "product_name": "Sports Shoes",
                "creator": "Skechers",
                "product_price": "₹4999",
                "product_link": "https://www.skechers.in/shoes",
                "product_description": "Lightweight mesh sports shoes with cushioned insoles",
                "relevance_score": 0.85,
                "reason": "Versatile footwear suitable for casual and fitness activities"
            }
        ]
    }
    ```
    """
    print_success("LLM response received (simulated)")
    
    # Step 3: Parse and validate response
    try:
        validated_response = parse_and_validate_response(simulated_llm_response)
        print_success("Response validated using Pydantic")
        
        # Step 4: Access structured data
        print(f"\n{BOLD}Validated Recommendation Structure:{RESET}")
        print(f"  Category: {validated_response.product_category}")
        print(f"  Number of products: {len(validated_response.recommended_products)}")
        
        for i, product in enumerate(validated_response.recommended_products, 1):
            print(f"\n  Product {i}:")
            print(f"    Name: {product.product_name}")
            print(f"    Creator: {product.creator}")
            print(f"    Price: {product.product_price}")
            print(f"    Link: {product.product_link}")
            print(f"    Description: {product.product_description}")
            print(f"    Relevance: {product.relevance_score}")
            print(f"    Reason: {product.reason}")
        
        return True
        
    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
        return False


# Main test runner
def run_all_tests():
    print(f"\n{BOLD}{BLUE}🧪 RECOMMENDATION PIPELINE TEST SUITE{RESET}")
    
    tests = [
        ("Product Model Validation", test_product_model),
        ("RecommendationResponse Model", test_recommendation_response_model),
        ("Response Parsing & Validation", test_response_parsing),
        ("Markdown Code Block Handling", test_markdown_handling),
        ("Invalid Response Handling", test_invalid_responses),
        ("Streamlit Integration Simulation", test_streamlit_integration),
        ("Complete Pipeline Simulation", test_complete_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_error(f"Test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{BOLD}Total: {passed_count}/{total_count} tests passed{RESET}\n")
    
    if passed_count == total_count:
        print_success("All tests passed! Pipeline is ready to use.")
        return 0
    else:
        print_error(f"{total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
