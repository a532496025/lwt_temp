import pytest
from guardrails import Guard
from pydantic import BaseModel, Field
from validator import LLMCritic


# Create a pydantic model with a field that uses the custom validator
class ValidatorTestObject(BaseModel):
    text: str = Field(
        validators=[
            LLMCritic(
                metrics={
                    "informative": {
                        "description": "An informative summary captures the main points of the input and is free of irrelevant details.",
                        "threshold": 75,
                    },
                    "coherent": {
                        "description": "A coherent summary is logically organized and easy to follow.",
                        "threshold": 50,
                    },
                    "concise": {
                        "description": "A concise summary is free of unnecessary repetition and wordiness.",
                        "threshold": 50,
                    },
                    "engaging": {
                        "description": "An engaging summary is interesting and holds the reader's attention.",
                        "threshold": 50,
                    },
                },
                max_score=100,
                llm_callable="gpt-3.5-turbo-0125",
                on_fail="exception",
            )
        ]
    )


# Create the guard object
guard = Guard.from_pydantic(output_class=ValidatorTestObject)


# Test happy path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "A judge has ordered former President Donald Trump to pay approximately $450 million to New York State in a civil 
            fraud case, which could significantly impact his financial assets. The ruling also restricts Trump from running any
            New York company and obtaining loans from New York banks for a specified period. These measures are described as 
            unprecedented threats to Trump's finances and may temporarily set back his real estate company. A court-appointed 
            monitor will oversee the family business. Trump's lawyer criticized the ruling, while these penalties could 
            foreshadow challenges he will face in upcoming criminal trials, which carry the potential for imprisonment."
        }
        """,
    ],
)
def test_happy_path(value):
    """Test happy path."""
    response = guard.parse(value)
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "Donald Trump was fined. No idea why."
        }
        """,
    ],
)
def test_fail_path(value):
    """Test fail path."""
    with pytest.raises(Exception):
        response = guard.parse(
            value,
        )
        print("Fail path response", response)
