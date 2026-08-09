# Product Recommendation with Agent

An AI-powered skincare recommendation system that uses facial analysis to predict demographics (age, gender, race) and provides personalized product recommendations through an agentic research pipeline.

## Features

- **Facial Analysis**: Predict age, gender, and ethnicity from a photo using deep learning models
- **Agentic Research**: Automated pipeline that researches skin concerns, finds products, and validates safety
- **Product Discovery**: Searches multiple sources (aistack, Open Beauty Facts, SerpAPI) for real products
- **Safety Gate**: LLM-based safety evaluation for pregnancy, allergies, and sensitivities
- **Configurable**: All settings via `config/api_config.yml` - prompts, models, timeouts, limits
- **Web UI**: Streamlit-based interface with photo upload, questionnaire, and formatted results

## Architecture

```
User Photo → Demographics Prediction → Questionnaire → Agentic Pipeline → Results
                                                    ↓
                                            ┌───────────────┐
                                            │ research_skin  │
                                            │ _concerns     │
                                            └───────┬───────┘
                                                    ↓
                                            ┌───────────────┐
                                            │ research_     │
                                            │ ingredients   │
                                            └───────┬───────┘
                                                    ↓
                                            ┌───────────────┐
                                            │ find_products │
                                            │ (3 sources)   │
                                            └───────┬───────┘
                                                    ↓
                                            ┌───────────────┐
                                            │ safety_gate   │
                                            │ (LLM agent)   │
                                            └───────┬───────┘
                                                    ↓
                                              Final Routine
```

## Project Structure

```
├── config/
│   └── api_config.yml          # All configuration (providers, prompts, timeouts)
├── src/
│   └── project_folder/
│       ├── agentic/            # Core recommendation engine
│       │   ├── __init__.py
│       │   ├── models.py       # Pydantic models for LLM outputs
│       │   ├── orchestrator.py # LangGraph state machine
│       │   ├── products.py     # Product discovery (3 sources)
│       │   ├── research.py     # Research sub-agents
│       │   ├── safety.py       # LLM safety gate
│       │   ├── providers.py    # LLM provider router with failover
│       │   ├── session.py      # User profile builder
│       │   ├── tools.py        # LangChain tools for planner
│       │   ├── config.py       # Config loader
│       │   ├── aistack.py      # Aistack search/crawl client
│       │   └── questionnaire.py # Static questionnaire
│       ├── app.py              # Streamlit UI
│       └── notebooks/          # Jupyter notebooks for model training
├── scripts/
│   └── api_runner.py           # CLI test harness
├── tests/
│   └── agentic/                # Unit tests (37 tests)
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.11.14
- API keys for LLM providers (at least one of: Agnes, OpenCode, LLM7IO, OracleLLM)
- SerpAPI key (optional, for product images and prices)

### Installation

```bash
# Clone the repository
git clone https://github.com/Prithwijit24/product_recommendation_with_agent.git
cd product_recommendation_with_agent

# Install dependencies
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```env
# LLM Providers (at least one required)
AGNES_API_KEY=your_key_here
OPENCODE_API_KEY=your_key_here
LLM7IO_API_KEY=your_key_here
ORACLELLM_API_KEY=your_key_here

# Product Search
SERP_API_KEY=your_serpapi_key        # Optional, for images/prices
AISTACK_BASE_URL=your_aistack_url    # Optional
AISTACK_API_KEY=your_aistack_key     # Optional

# LangSmith (optional, for tracing)
LANGSMITH_API_KEY=your_key_here
```

### Run the Web UI

```bash
streamlit run src/project_folder/app.py
```

### Run CLI Test

```bash
python scripts/api_runner.py --preset app-f
python scripts/api_runner.py --preset app-m
python scripts/api_runner.py --race asian --age-range 25-34 --sex F
```

## Configuration

All settings are in `config/api_config.yml`:

```yaml
providers:
  agnes:
    base_url: https://apihub.agnes-ai.com/v1
    planner_model: agnes-2.0-flash
    worker_model: agnes-2.0-flash

orchestrator:
  max_turns: 14
  max_safety_retries: 2
  routine_size: 3
  recursion_limit: 80

products:
  default_budget: medium
  budget_limits:
    low: 30.0
    medium: 90.0
    high: .inf
```

## Product Discovery Pipeline

The system searches for products in this order:

1. **Aistack Search** (free/cheap) - Primary source
2. **Open Beauty Facts** (free) - Fallback
3. **SerpAPI** (costly) - Final fallback, includes images and prices

For products without images, the system fetches the product page and extracts images using:
- Open Graph (`og:image`) meta tags
- Schema.org image markup
- Direct product image URLs (for Open Beauty Facts)

## Safety Gate

The LLM safety agent evaluates products against:
- Pregnancy contraindications (tretinoin, retinol, isotretinoin, salicylic acid, benzoyl peroxide)
- Fragrance sensitivities
- General skin sensitivity

Products that fail the safety check are removed or replaced (up to 2 retries).

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/agentic/test_products.py -v
```

37 unit tests covering:
- Product discovery and filtering
- Research sub-agents
- Safety gate
- Provider router
- Orchestrator state machine

## API Reference

### Main Entry Point

```python
from project_folder.agentic import orchestrate

result = orchestrate(
    demographics={"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Asian"}},
    answers={"skin_type": "Combination", "budget": "medium", "pregnant": "no"},
    session_id="unique-id"
)

# Returns:
# {
#     "session_id": "unique-id",
#     "routine": [
#         {
#             "product_name": "...",
#             "url": "...",
#             "price": "$XX.XX",
#             "ingredient": "...",
#             "reasoning": "...",
#             "image_url": "..."
#         }
#     ],
#     "concerns_addressed": ["..."],
#     "disclaimer": "..."
# }
```

## Technologies

- **LangGraph**: Agent state machine
- **LangChain**: LLM framework
- **Pydantic**: Data validation
- **Streamlit**: Web UI
- **SerpAPI**: Product search (Google Shopping)
- **TensorFlow/Keras**: Facial analysis models

## License

Private project - All rights reserved.
