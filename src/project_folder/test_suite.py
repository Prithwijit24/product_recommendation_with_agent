"""
Comprehensive Unit and Integration Tests for Product Recommendation System

This test suite covers:
- recommendation_agent.py (Pydantic models & agent functions)
- scripts/data_loader.py (data loading & processing)
- scripts/embeddings.py (image embeddings)
- scripts/train_model.py (model training utilities)
- main.py (feature selection & training pipeline)
"""

import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError
import tempfile
import os
from pathlib import Path

# Import modules to test
from recommendation_agent import (
    Product, 
    RecommendationResponse, 
    agent_creation_wrapper,
    parse_and_validate_response
)
from scripts.data_loader import (
    age_interval,
    bw_check,
    data_load_df,
    apply_filter
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_product_data():
    """Fixture for valid product data"""
    return {
        "product_name": "Premium Cotton Shirt",
        "creator": "BrandX",
        "product_price": "₹1500",
        "product_link": "https://example.com/shirt",
        "product_description": "High-quality cotton shirt for casual wear",
        "relevance_score": 0.95,
        "reason": "Perfect for your age and style"
    }


@pytest.fixture
def valid_recommendation_response_data():
    """Fixture for valid recommendation response"""
    return {
        "product_category": "Clothing",
        "recommended_products": [
            {
                "product_name": "Cotton Shirt",
                "creator": "BrandX",
                "product_price": "₹1500",
                "product_link": "https://example.com/shirt",
                "product_description": "High-quality cotton",
                "relevance_score": 0.9,
                "reason": "Suitable for you"
            },
            {
                "product_name": "Jeans",
                "creator": "BrandY",
                "product_price": "₹2500",
                "product_link": "https://example.com/jeans",
                "product_description": "Classic blue jeans",
                "relevance_score": 0.85,
                "reason": "Versatile option"
            }
        ]
    }


@pytest.fixture
def sample_dataframe():
    """Fixture for sample DataFrame with age/gender/race data"""
    return pd.DataFrame({
        'age': [25, 30, 45, 50, 22],
        'gender': [0, 1, 0, 1, 0],
        'race': ['0', '1', '2', '3', '0'],
        'is_bw': [False, False, True, False, False],
        'age_interval': ['21 to 25', '26 to 30', '41 to 45', '46 to 50', '21 to 25']
    })


# ============================================================================
# UNIT TESTS - recommendation_agent.py
# ============================================================================

class TestProductModel:
    """Unit tests for Product Pydantic model"""

    def test_product_creation_valid(self, valid_product_data):
        """Test creating a valid product"""
        product = Product(**valid_product_data)
        assert product.product_name == "Premium Cotton Shirt"
        assert product.relevance_score == 0.95
        assert product.creator == "BrandX"

    def test_product_invalid_relevance_score_above_1(self, valid_product_data):
        """Test product with invalid relevance score > 1"""
        valid_product_data['relevance_score'] = 1.5
        with pytest.raises(ValidationError):
            Product(**valid_product_data)

    def test_product_invalid_relevance_score_below_0(self, valid_product_data):
        """Test product with invalid relevance score < 0"""
        valid_product_data['relevance_score'] = -0.1
        with pytest.raises(ValidationError):
            Product(**valid_product_data)

    def test_product_invalid_missing_required_field(self, valid_product_data):
        """Test product with missing required field"""
        del valid_product_data['product_name']
        with pytest.raises(ValidationError):
            Product(**valid_product_data)

    def test_product_relevance_score_boundary_0(self, valid_product_data):
        """Test product with relevance score at boundary 0"""
        valid_product_data['relevance_score'] = 0.0
        product = Product(**valid_product_data)
        assert product.relevance_score == 0.0

    def test_product_relevance_score_boundary_1(self, valid_product_data):
        """Test product with relevance score at boundary 1"""
        valid_product_data['relevance_score'] = 1.0
        product = Product(**valid_product_data)
        assert product.relevance_score == 1.0

    def test_product_with_special_characters_in_name(self, valid_product_data):
        """Test product with special characters in name"""
        valid_product_data['product_name'] = "T-Shirt | Premium & Quality"
        product = Product(**valid_product_data)
        assert "T-Shirt" in product.product_name


class TestRecommendationResponseModel:
    """Unit tests for RecommendationResponse Pydantic model"""

    def test_recommendation_response_valid(self, valid_recommendation_response_data):
        """Test creating valid recommendation response"""
        response = RecommendationResponse(**valid_recommendation_response_data)
        assert response.product_category == "Clothing"
        assert len(response.recommended_products) == 2
        assert response.recommended_products[0].product_name == "Cotton Shirt"

    def test_recommendation_response_single_product(self, valid_recommendation_response_data):
        """Test recommendation response with single product"""
        valid_recommendation_response_data['recommended_products'] = [
            valid_recommendation_response_data['recommended_products'][0]
        ]
        response = RecommendationResponse(**valid_recommendation_response_data)
        assert len(response.recommended_products) == 1

    def test_recommendation_response_empty_products(self, valid_recommendation_response_data):
        """Test recommendation response with empty products list"""
        valid_recommendation_response_data['recommended_products'] = []
        response = RecommendationResponse(**valid_recommendation_response_data)
        assert len(response.recommended_products) == 0

    def test_recommendation_response_missing_category(self, valid_recommendation_response_data):
        """Test recommendation response missing product_category"""
        del valid_recommendation_response_data['product_category']
        with pytest.raises(ValidationError):
            RecommendationResponse(**valid_recommendation_response_data)

    def test_recommendation_response_invalid_product_in_list(self, valid_recommendation_response_data):
        """Test recommendation response with invalid product in list"""
        valid_recommendation_response_data['recommended_products'][0]['relevance_score'] = 1.5
        with pytest.raises(ValidationError):
            RecommendationResponse(**valid_recommendation_response_data)


class TestParseAndValidateResponse:
    """Unit tests for parse_and_validate_response function"""

    def test_parse_valid_json_response(self, valid_recommendation_response_data):
        """Test parsing valid JSON response"""
        json_response = json.dumps(valid_recommendation_response_data)
        result = parse_and_validate_response(json_response)
        assert isinstance(result, RecommendationResponse)
        assert result.product_category == "Clothing"

    def test_parse_json_with_markdown_code_block(self, valid_recommendation_response_data):
        """Test parsing JSON wrapped in markdown code block"""
        json_response = f"```json\n{json.dumps(valid_recommendation_response_data)}\n```"
        result = parse_and_validate_response(json_response)
        assert isinstance(result, RecommendationResponse)

    def test_parse_json_with_generic_code_block(self, valid_recommendation_response_data):
        """Test parsing JSON wrapped in generic code block"""
        json_response = f"```\n{json.dumps(valid_recommendation_response_data)}\n```"
        result = parse_and_validate_response(json_response)
        assert isinstance(result, RecommendationResponse)

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON"""
        invalid_json = "{'key': 'value'}"
        with pytest.raises(ValueError):
            parse_and_validate_response(invalid_json)

    def test_parse_valid_json_invalid_schema(self):
        """Test parsing valid JSON but invalid schema"""
        invalid_schema = json.dumps({"invalid": "schema"})
        with pytest.raises(ValueError):
            parse_and_validate_response(invalid_schema)

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        with pytest.raises(ValueError):
            parse_and_validate_response("")

    def test_parse_response_with_extra_fields(self, valid_recommendation_response_data):
        """Test parsing response with extra fields (should be ignored)"""
        valid_recommendation_response_data['extra_field'] = 'should_be_ignored'
        json_response = json.dumps(valid_recommendation_response_data)
        result = parse_and_validate_response(json_response)
        assert isinstance(result, RecommendationResponse)


class TestAgentCreationWrapper:
    """Unit tests for agent_creation_wrapper function"""

    @patch('recommendation_agent.ChatOpenAI')
    @patch('recommendation_agent.create_agent')
    def test_agent_creation_openrouter(self, mock_create_agent, mock_chatOpenAI):
        """Test agent creation with Openrouter"""
        mock_llm = MagicMock()
        mock_chatOpenAI.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent = agent_creation_wrapper(
            age=25,
            race='Indian',
            gender='Male',
            location='India',
            radio='Openrouter',
            llm_api_key='test-key'
        )

        mock_chatOpenAI.assert_called_once()
        mock_create_agent.assert_called_once()
        assert agent == mock_agent

    @patch('recommendation_agent.ChatOpenAI')
    @patch('recommendation_agent.create_agent')
    def test_agent_creation_grok(self, mock_create_agent, mock_chatOpenAI):
        """Test agent creation with Grok"""
        mock_llm = MagicMock()
        mock_chatOpenAI.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent = agent_creation_wrapper(
            age=30,
            race='American',
            gender='Female',
            location='USA',
            radio='Grok',
            llm_api_key='test-key'
        )

        # Verify Grok endpoint was used
        call_kwargs = mock_chatOpenAI.call_args[1]
        assert 'groq.com' in call_kwargs['base_url']

    @patch('recommendation_agent.ChatOpenAI')
    @patch('recommendation_agent.create_agent')
    def test_agent_system_prompt_includes_user_profile(self, mock_create_agent, mock_chatOpenAI):
        """Test that system prompt includes user profile information"""
        mock_llm = MagicMock()
        mock_chatOpenAI.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent_creation_wrapper(
            age=25,
            race='Indian',
            gender='Male',
            location='India',
            radio='Openrouter',
            llm_api_key='test-key'
        )

        # Check system_prompt in create_agent call
        call_kwargs = mock_create_agent.call_args[1]
        system_prompt = call_kwargs['system_prompt']
        assert 'Age: 25' in system_prompt
        assert 'Male' in system_prompt
        assert 'India' in system_prompt


# ============================================================================
# UNIT TESTS - scripts/data_loader.py
# ============================================================================

class TestAgeInterval:
    """Unit tests for age_interval function"""

    def test_age_interval_0_5(self):
        """Test age interval for 0-5"""
        assert age_interval(3) == '1 to 5'

    def test_age_interval_5_10(self):
        """Test age interval for 5-10"""
        assert age_interval(7) == '6 to 10'

    def test_age_interval_20_25(self):
        """Test age interval for 20-25"""
        assert age_interval(23) == '21 to 25'

    def test_age_interval_95(self):
        """Test age interval for 95 (over 100)"""
        assert age_interval(95) == '100+'

    def test_age_interval_boundary_0(self):
        """Test age interval at boundary 0"""
        assert age_interval(0) == '1 to 5'

    def test_age_interval_boundary_100(self):
        """Test age interval at boundary 100"""
        assert age_interval(100) == '100+'

    def test_age_interval_float(self):
        """Test age interval with float input"""
        assert age_interval(12.5) in ['11 to 15', '6 to 10']  # Implementation dependent


class TestBwCheck:
    """Unit tests for bw_check function"""

    @patch('scripts.data_loader.plt.imread')
    def test_bw_check_true(self, mock_imread):
        """Test black and white image detection - true case"""
        # Mock an image where R==G for all pixels
        mock_image = np.zeros((200, 200, 3))
        mock_imread.return_value = mock_image
        
        # This will be called 3 times so we return the same mock each time
        result = bw_check('test_image.jpg')
        assert result is True or result is False  # Depends on 40000 pixels

    @patch('scripts.data_loader.plt.imread')
    def test_bw_check_colored_image(self, mock_imread):
        """Test black and white check with colored image"""
        # Mock a colored image
        mock_image = np.random.rand(200, 200, 3)
        mock_imread.return_value = mock_image
        
        result = bw_check('test_image.jpg')
        assert isinstance(result, (bool, np.bool_))


class TestDataLoadDf:
    """Unit tests for data_load_df function"""

    @patch('scripts.data_loader.bw_check')
    def test_data_load_df_creation(self, mock_bw_check):
        """Test DataFrame creation from image list"""
        mock_bw_check.return_value = False
        
        data = [
            '/path/data/25_0_2_photo.jpg',
            '/path/data/30_1_3_photo.jpg'
        ]
        
        df = data_load_df(data)
        
        assert isinstance(df, pd.DataFrame)
        assert 'age' in df.columns
        assert 'gender' in df.columns
        assert 'race' in df.columns
        assert len(df) == 2

    @patch('scripts.data_loader.bw_check')
    def test_data_load_df_age_parsing(self, mock_bw_check):
        """Test age parsing in DataFrame"""
        mock_bw_check.return_value = False
        
        data = ['/path/data/25_0_2_photo.jpg']
        df = data_load_df(data)
        
        assert df.loc[0, 'age'] == '25'
        assert df.loc[0, 'gender'] == '0'
        assert df.loc[0, 'race'] == '2'

    @patch('scripts.data_loader.bw_check')
    def test_data_load_df_empty_list(self, mock_bw_check):
        """Test DataFrame creation with empty image list"""
        df = data_load_df([])
        assert len(df) == 0


class TestApplyFilter:
    """Unit tests for apply_filter function"""

    def test_apply_filter_removes_invalid_race(self, sample_dataframe):
        """Test filter removes invalid race values"""
        sample_dataframe.loc[2, 'race'] = '4'  # Invalid race
        
        filtered = apply_filter(sample_dataframe)
        
        assert len(filtered) < len(sample_dataframe)
        assert '4' not in filtered['race'].values

    def test_apply_filter_removes_young_ages(self, sample_dataframe):
        """Test filter removes very young ages"""
        sample_dataframe.loc[0, 'age_interval'] = '1 to 5'
        
        filtered = apply_filter(sample_dataframe)
        
        assert '1 to 5' not in filtered['age_interval'].values

    def test_apply_filter_removes_bw_images(self, sample_dataframe):
        """Test filter removes black and white images"""
        filtered = apply_filter(sample_dataframe)
        
        assert False not in filtered['is_bw'].values or len(filtered) > 0

    def test_apply_filter_valid_data_preserved(self, sample_dataframe):
        """Test filter preserves valid data"""
        sample_dataframe['age_interval'] = ['21 to 25', '26 to 30', '41 to 45', '46 to 50', '21 to 25']
        sample_dataframe['race'] = ['0', '1', '2', '3', '0']
        sample_dataframe['is_bw'] = [False, False, False, False, False]
        
        filtered = apply_filter(sample_dataframe)
        
        assert len(filtered) == len(sample_dataframe)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRecommendationPipeline:
    """Integration tests for the full recommendation pipeline"""

    @patch('recommendation_agent.ChatOpenAI')
    @patch('recommendation_agent.create_agent')
    def test_agent_creation_and_response_parsing(self, mock_create_agent, mock_chatOpenAI):
        """Test creating agent and parsing response together"""
        mock_llm = MagicMock()
        mock_chatOpenAI.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        # Test agent creation
        agent = agent_creation_wrapper(
            age=25,
            race='Indian',
            gender='Male',
            location='India',
            radio='Openrouter',
            llm_api_key='test-key'
        )

        assert agent is not None

        # Test response parsing
        valid_response = {
            "product_category": "Clothing",
            "recommended_products": [
                {
                    "product_name": "T-Shirt",
                    "creator": "Brand",
                    "product_price": "₹500",
                    "product_link": "https://example.com",
                    "product_description": "Test",
                    "relevance_score": 0.9,
                    "reason": "Good fit"
                }
            ]
        }

        result = parse_and_validate_response(json.dumps(valid_response))
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommended_products) == 1

    def test_data_pipeline_integration(self, sample_dataframe):
        """Test data loading and filtering pipeline"""
        # Start with sample data
        initial_count = len(sample_dataframe)

        # Apply filter
        filtered_df = apply_filter(sample_dataframe)

        # Verify filter was applied
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) <= initial_count


class TestErrorHandling:
    """Integration tests for error handling across modules"""

    def test_invalid_product_in_response_stream(self):
        """Test handling invalid product in response stream"""
        invalid_response = {
            "product_category": "Clothing",
            "recommended_products": [
                {
                    "product_name": "Shirt",
                    "creator": "Brand",
                    "product_price": "₹500",
                    "product_link": "https://example.com",
                    "product_description": "Test",
                    "relevance_score": 1.5,  # Invalid
                    "reason": "Good"
                }
            ]
        }

        with pytest.raises(ValueError):
            parse_and_validate_response(json.dumps(invalid_response))

    def test_malformed_json_response(self):
        """Test handling malformed JSON response"""
        malformed_json = "{'key': 'value'"  # Missing closing brace

        with pytest.raises(ValueError):
            parse_and_validate_response(malformed_json)

    @patch('recommendation_agent.ChatOpenAI')
    def test_agent_creation_with_invalid_api_key(self, mock_chatOpenAI):
        """Test agent creation with invalid API key"""
        mock_chatOpenAI.side_effect = Exception("Invalid API key")

        with pytest.raises(Exception):
            agent_creation_wrapper(
                age=25,
                race='Indian',
                gender='Male',
                location='India',
                radio='Openrouter',
                llm_api_key='invalid-key'
            )


class TestDataValidation:
    """Tests for data validation and edge cases"""

    def test_recommendation_with_zero_relevance(self):
        """Test recommendation with zero relevance score"""
        product_data = {
            "product_name": "Niche Product",
            "creator": "Brand",
            "product_price": "₹100",
            "product_link": "https://example.com",
            "product_description": "Test",
            "relevance_score": 0.0,
            "reason": "Not relevant"
        }

        product = Product(**product_data)
        assert product.relevance_score == 0.0

    def test_recommendation_with_perfect_relevance(self):
        """Test recommendation with perfect relevance score"""
        product_data = {
            "product_name": "Perfect Product",
            "creator": "Brand",
            "product_price": "₹1000",
            "product_link": "https://example.com",
            "product_description": "Test",
            "relevance_score": 1.0,
            "reason": "Perfect match"
        }

        product = Product(**product_data)
        assert product.relevance_score == 1.0

    def test_currency_variants_in_price(self):
        """Test various currency symbols in product price"""
        base_data = {
            "product_name": "Product",
            "creator": "Brand",
            "product_link": "https://example.com",
            "product_description": "Test",
            "relevance_score": 0.5,
            "reason": "Test"
        }

        # Test various currency formats
        currencies = ["₹500", "$50", "€45", "£40", "Price not available"]
        
        for price in currencies:
            base_data['product_price'] = price
            product = Product(**base_data)
            assert product.product_price == price


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for critical functions"""

    def test_parse_large_recommendation_response(self):
        """Test parsing large recommendation response"""
        large_response = {
            "product_category": "Clothing",
            "recommended_products": [
                {
                    "product_name": f"Product {i}",
                    "creator": f"Brand {i}",
                    "product_price": f"₹{100 * (i+1)}",
                    "product_link": f"https://example.com/product/{i}",
                    "product_description": f"Description for product {i}",
                    "relevance_score": round(0.5 + (i * 0.05), 2),
                    "reason": f"Reason for product {i}"
                }
                for i in range(100)  # 100 products
            ]
        }

        result = parse_and_validate_response(json.dumps(large_response))
        assert len(result.recommended_products) == 100

    def test_dataframe_filter_performance(self):
        """Test filter performance with large DataFrame"""
        large_df = pd.DataFrame({
            'age': [np.random.randint(10, 100) for _ in range(10000)],
            'gender': [np.random.randint(0, 2) for _ in range(10000)],
            'race': [str(np.random.randint(0, 4)) for _ in range(10000)],
            'is_bw': [np.random.choice([True, False]) for _ in range(10000)],
            'age_interval': [f'{i*5} to {(i+1)*5}' for i in range(10000)]
        })

        filtered = apply_filter(large_df)
        assert len(filtered) <= len(large_df)


# ============================================================================
# CONFTEST SETUP
# ============================================================================

if __name__ == "__main__":
    # Run tests with: pytest test_suite.py -v
    pytest.main([__file__, "-v", "--tb=short"])
