"""
Comprehensive tests for optimizer backbone components (Chat, UserTurn, AssistantTurn)
Tests include: truncation strategies, multimodal content, and conversation management

We need to test a few things:
1. Various use cases of ContentBlock and specialized ones
2. UserTurn, AssistantTurn and conversation manager
3. Multi-modal use of conversation manager, including multi-turn and image as output
"""
import os
import base64
import pytest
from opto.utils.backbone import (
    Chat,
    UserTurn,
    AssistantTurn
)

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get(
    "OPENAI_API_KEY")


# ============================================================================
# Test Fixtures
# ============================================================================

def create_sample_conversation():
    """Create a sample conversation with multiple rounds"""
    history = Chat(system_prompt="You are a helpful assistant.")
    
    # Round 1
    user1 = UserTurn().add_text("Hello, what's the weather?")
    assistant1 = AssistantTurn().add_text("The weather is sunny today.")
    history.add_user_turn(user1).add_assistant_turn(assistant1)
    
    # Round 2
    user2 = UserTurn().add_text("What about tomorrow?")
    assistant2 = AssistantTurn().add_text("Tomorrow will be rainy.")
    history.add_user_turn(user2).add_assistant_turn(assistant2)
    
    # Round 3
    user3 = UserTurn().add_text("Should I bring an umbrella?")
    assistant3 = AssistantTurn().add_text("Yes, definitely bring an umbrella.")
    history.add_user_turn(user3).add_assistant_turn(assistant3)
    
    # Round 4
    user4 = UserTurn().add_text("Thanks for the advice!")
    assistant4 = AssistantTurn().add_text("You're welcome! Stay dry!")
    history.add_user_turn(user4).add_assistant_turn(assistant4)
    
    return history


# ============================================================================
# Truncation Tests
# ============================================================================

def test_default_all_history():
    """Test default behavior (n=-1) returns all history"""
    history = create_sample_conversation()
    
    messages = history.to_messages()
    
    # Should have: system + 8 turns (4 user + 4 assistant)
    assert len(messages) == 9  # 1 system + 8 messages
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[-1]["role"] == "assistant"


def test_truncate_from_start():
    """Test truncate_from_start strategy - keeps last N rounds"""
    history = create_sample_conversation()
    
    # Keep last 2 rounds (4 turns)
    messages = history.to_messages(n=2, truncate_strategy="from_start")
    
    # Should have: system + 2 rounds (4 turns)
    assert len(messages) == 5  # 1 system + 4 messages
    assert messages[0]["role"] == "system"
    
    # Should have the last 2 rounds (round 3 and round 4)
    # Round 3: user3 (umbrella question), assistant3 (umbrella answer)
    # Round 4: user4 (thanks), assistant4 (welcome)
    assert messages[1]["role"] == "user"
    assert "umbrella" in messages[1]["content"][0]["text"]
    assert messages[2]["role"] == "assistant"
    # Content is now a list of dicts with type and text fields
    assert any("umbrella" in item.get("text", "") for item in messages[2]["content"])
    assert messages[3]["role"] == "user"
    assert "Thanks" in messages[3]["content"][0]["text"]
    assert messages[4]["role"] == "assistant"
    # Content is now a list of dicts with type and text fields
    assert any("welcome" in item.get("text", "") for item in messages[4]["content"])


def test_truncate_from_end():
    """Test truncate_from_end strategy - keeps first N rounds"""
    history = create_sample_conversation()
    
    # Keep first 2 rounds (4 turns)
    messages = history.to_messages(n=2, truncate_strategy="from_end")
    
    # Should have: system + 2 rounds (4 turns)
    assert len(messages) == 5  # 1 system + 4 messages
    assert messages[0]["role"] == "system"
    
    # Should have the first 2 rounds (round 1 and round 2)
    # Round 1: user1 (weather), assistant1 (sunny)
    # Round 2: user2 (tomorrow), assistant2 (rainy)
    assert messages[1]["role"] == "user"
    assert "Hello" in messages[1]["content"][0]["text"]
    assert messages[2]["role"] == "assistant"
    # Content is now a list of dicts with type and text fields
    assert any("sunny" in item.get("text", "") for item in messages[2]["content"])
    assert messages[3]["role"] == "user"
    assert "tomorrow" in messages[3]["content"][0]["text"]
    assert messages[4]["role"] == "assistant"
    # Content is now a list of dicts with type and text fields
    assert any("rainy" in item.get("text", "") for item in messages[4]["content"])


def test_truncate_zero_turns():
    """Test truncating to 0 turns"""
    history = create_sample_conversation()
    
    messages = history.to_messages(n=0, truncate_strategy="from_start")
    
    # Should only have system message
    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_truncate_more_than_available():
    """Test requesting more turns than available"""
    history = create_sample_conversation()
    
    # Request 100 turns but only have 8
    messages = history.to_messages(n=100, truncate_strategy="from_start")
    
    # Should return all available
    assert len(messages) == 9  # 1 system + 8 messages


def test_empty_conversation():
    """Test truncation on empty conversation"""
    history = Chat(system_prompt="Test")
    
    messages = history.to_messages(n=5)
    
    assert len(messages) == 1  # Just system
    assert messages[0]["role"] == "system"


def test_to_litellm_format_with_truncation():
    """Test to_litellm_format() also supports truncation"""
    history = create_sample_conversation()
    
    # n=2 means 2 rounds (4 turns), from_end keeps first 2 rounds
    messages = history.to_litellm_format(n=2, truncate_strategy="from_end")
    
    # Should have: system + 2 rounds (4 turns)
    assert len(messages) == 5
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[4]["role"] == "assistant"


def test_invalid_strategy():
    """Test that invalid strategy raises error"""
    history = create_sample_conversation()
    
    with pytest.raises(ValueError, match="Unknown truncate_strategy"):
        history.to_messages(n=2, truncate_strategy="invalid_strategy")


def test_negative_n_values():
    """Test that n=-1 returns all history"""
    history = create_sample_conversation()
    
    # n=-1 should return all
    messages_all = history.to_messages(n=-1)
    assert len(messages_all) == 9
    
    # Verify it's the same as not passing n at all
    messages_default = history.to_messages()
    assert len(messages_all) == len(messages_default)


# ============================================================================
# Multimodal / Multi-Image Tests
# ============================================================================

def test_user_turn_multiple_images():
    """Test that a user turn can have multiple images"""
    history = Chat()
    
    # Create a user turn with text and multiple images (like the OpenAI example)
    user_turn = (UserTurn()
                 .add_text("What are in these images? Is there any difference between them?")
                 .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                 .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))
    
    history.add_user_turn(user_turn)
    
    # Convert to LiteLLM format
    messages = history.to_litellm_format()
    
    # Should have 1 message
    assert len(messages) == 1
    
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    
    # Content should be a list with 3 items: 1 text + 2 images
    assert len(user_msg["content"]) == 3
    
    # Check first item is text
    assert user_msg["content"][0]["type"] == "input_text"
    assert user_msg["content"][0]["text"] == "What are in these images? Is there any difference between them?"
    
    # Check second item is first image
    assert user_msg["content"][1]["type"] == "input_image"
    assert user_msg["content"][1]["image_url"] == "https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg"
    
    # Check third item is second image
    assert user_msg["content"][2]["type"] == "input_image"
    assert user_msg["content"][2]["image_url"] == "https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"


def test_assistant_turn_multiple_images():
    """Test that an assistant turn can also have multiple images (for models that generate images)"""
    history = Chat()
    
    # Assistant turn with text and multiple images
    assistant_turn = (AssistantTurn()
                      .add_text("Here are two generated images based on your request:")
                      .add_image(url="https://example.com/generated1.png")
                      .add_image(url="https://example.com/generated2.png"))
    
    history.add_assistant_turn(assistant_turn)
    
    # Convert to LiteLLM format
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    
    # Assistant should have text content (now in list format)
    assert any("Here are two generated images" in item.get("text", "") for item in messages[0]["content"])


def test_mixed_content_types_in_turn():
    """Test mixing text, images, and other content types in a single turn"""
    history = Chat()
    
    # Create a complex turn with multiple content types
    user_turn = (UserTurn()
                 .add_text("Please analyze these images and this document:")
                 .add_image(url="https://example.com/chart1.png")
                 .add_image(url="https://example.com/chart2.png")
                 .add_text("What patterns do you see?"))
    
    history.add_user_turn(user_turn)
    
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    user_msg = messages[0]
    
    # Should have 4 content blocks: text, image, image, text
    assert len(user_msg["content"]) == 4
    assert user_msg["content"][0]["type"] == "input_text"
    assert user_msg["content"][1]["type"] == "input_image"
    assert user_msg["content"][2]["type"] == "input_image"
    assert user_msg["content"][3]["type"] == "input_text"


def test_multiple_images_with_base64():
    """Test multiple images using base64 encoding"""
    history = Chat()
    
    # Create fake base64 image data
    fake_image_data1 = base64.b64encode(b"fake image 1").decode('utf-8')
    fake_image_data2 = base64.b64encode(b"fake image 2").decode('utf-8')
    
    user_turn = (UserTurn()
                 .add_text("Compare these two images:")
                 .add_image(data=fake_image_data1, media_type="image/png")
                 .add_image(data=fake_image_data2, media_type="image/jpeg"))
    
    history.add_user_turn(user_turn)
    
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    user_msg = messages[0]
    
    # Should have 3 content blocks
    assert len(user_msg["content"]) == 3
    
    # Check base64 data URLs are properly formatted
    assert user_msg["content"][1]["type"] == "input_image"
    assert user_msg["content"][1]["image_url"].startswith("data:image/png;base64,")
    
    assert user_msg["content"][2]["type"] == "input_image"
    assert user_msg["content"][2]["image_url"].startswith("data:image/jpeg;base64,")


def test_conversation_with_multiple_multi_image_turns():
    """Test a full conversation where multiple turns each have multiple images"""
    history = Chat(system_prompt="You are a helpful image analysis assistant.")
    
    # User turn 1: Multiple images
    user1 = (UserTurn()
             .add_text("What's the difference between these flowers?")
             .add_image(url="https://example.com/rose.jpg")
             .add_image(url="https://example.com/tulip.jpg"))
    history.add_user_turn(user1)
    
    # Assistant response
    assistant1 = AssistantTurn().add_text("The first is a rose with layered petals, the second is a tulip with a cup shape.")
    history.add_assistant_turn(assistant1)
    
    # User turn 2: More images
    user2 = (UserTurn()
             .add_text("Now compare these landscapes:")
             .add_image(url="https://example.com/mountain.jpg")
             .add_image(url="https://example.com/beach.jpg")
             .add_image(url="https://example.com/forest.jpg"))
    history.add_user_turn(user2)
    
    messages = history.to_litellm_format()
    
    # Should have: system + user1 + assistant1 + user2
    assert len(messages) == 4
    
    # Check user1 has 3 content blocks (1 text + 2 images)
    assert len(messages[1]["content"]) == 3
    
    # Check user2 has 4 content blocks (1 text + 3 images)
    assert len(messages[3]["content"]) == 4


# ============================================================================
# Integration Tests - Truncation + Multimodal
# ============================================================================

def test_truncate_multimodal_conversation():
    """Test truncation works correctly with multimodal content"""
    history = Chat(system_prompt="You are a vision assistant.")
    
    # Add several turns with images (5 rounds = 10 turns)
    for i in range(5):
        user = (UserTurn()
                .add_text(f"Analyze image {i}")
                .add_image(url=f"https://example.com/image{i}.jpg"))
        assistant = AssistantTurn().add_text(f"Analysis of image {i}")
        history.add_user_turn(user).add_assistant_turn(assistant)
    
    # Truncate to last 2 rounds (4 turns)
    messages = history.to_messages(n=2, truncate_strategy="from_start")
    
    # Should have system + 2 rounds (4 turns)
    assert len(messages) == 5
    
    # Check that multimodal content is preserved
    assert len(messages[1]["content"]) == 2  # text + image
    assert messages[1]["content"][1]["type"] == "input_image"

# ============================================================================
# Real LLM Call Tests with Images
# ============================================================================

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_real_llm_call_with_multiple_images():
    """Test sending real images to GPT and getting a response.
    
    This test sends two flower images to GPT-4 Vision and asks it to compare them.
    """
    from opto.utils.llm import LLM
    
    # Create conversation with images
    history = Chat(system_prompt="You are a helpful assistant that can analyze images.")
    
    # Create a user turn with text and two real flower images
    user_turn = (UserTurn()
                 .add_text("What are in these images? Is there any difference between them? Please describe each image briefly.")
                 .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                 .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))
    
    history.add_user_turn(user_turn)
    
    # Get messages in LiteLLM format
    messages = history.to_litellm_format()
    
    print("\n" + "="*80)
    print("REAL LLM CALL WITH MULTIPLE IMAGES")
    print("="*80)
    print(f"\nSending {len(user_turn.content)} content blocks (1 text + 2 images)...")
    
    # Make the LLM call with mm_beta=True for Response API format
    llm = LLM(mm_beta=True)
    response = llm(messages=messages, max_tokens=500)
    
    # response is now an AssistantTurn object
    response_content = response.to_text()
    
    print("\n📷 User Query:")
    print("  What are in these images? Is there any difference between them?")
    print("\n🤖 GPT Response:")
    print("-" * 40)
    print(response_content)
    print("-" * 40)
    
    # Store assistant response in history
    history.add_assistant_turn(response)
    
    # Verify we got a meaningful response
    assert response_content is not None
    assert len(response_content) > 50  # Should have some substantial content
    
    # The response should mention something about flowers/images
    response_lower = response_content.lower()
    assert any(word in response_lower for word in ["flower", "image", "picture", "rose", "pink", "red", "petal"]), \
        f"Response doesn't seem to describe the flower images: {response_content[:200]}..."
    
    print("\n✅ Successfully received and validated GPT response about the images!")


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_real_llm_multi_turn_with_images():
    """Test a multi-turn conversation with images.
    
    First turn: Ask about images
    Second turn: Follow-up question about the same images
    """
    from opto.utils.llm import LLM
    
    history = Chat(system_prompt="You are a helpful assistant that can analyze images.")
    llm = LLM(mm_beta=True)
    
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION WITH IMAGES")
    print("="*80)
    
    # Turn 1: Send images and ask about them
    user_turn1 = (UserTurn()
                  .add_text("What type of flowers are shown in these images?")
                  .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                  .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))
    
    history.add_user_turn(user_turn1)
    messages = history.to_litellm_format()
    
    print("\n📷 Turn 1 - User:")
    print("  What type of flowers are shown in these images? [+ 2 images]")
    
    response1 = llm(messages=messages, max_tokens=300)
    response1_content = response1.to_text()
    
    print("\n🤖 Turn 1 - Assistant:")
    print(f"  {response1_content[:200]}...")
    
    history.add_assistant_turn(response1)
    
    # Turn 2: Follow-up question (no new images, but context from previous turn)
    user_turn2 = UserTurn().add_text("Which of these flowers would be better for a romantic gift and why?")
    history.add_user_turn(user_turn2)
    
    messages = history.to_litellm_format()
    
    print("\n📷 Turn 2 - User:")
    print("  Which of these flowers would be better for a romantic gift and why?")
    
    response2 = llm(messages=messages, max_tokens=300)
    response2_content = response2.to_text()
    
    print("\n🤖 Turn 2 - Assistant:")
    print(f"  {response2_content[:200]}...")
    
    # Verify responses
    assert response1_content is not None and len(response1_content) > 20
    assert response2_content is not None and len(response2_content) > 20
    
    # Turn 2 should reference the context from turn 1
    response2_lower = response2_content.lower()
    assert any(word in response2_lower for word in ["flower", "rose", "romantic", "gift", "love"]), \
        "Turn 2 response doesn't seem to reference the flower context"
    
    print("\n✅ Multi-turn conversation with images completed successfully!")


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_real_llm_multi_turn_with_images_updated_assistant_turn():
    """Test a multi-turn conversation with images.
    
    First turn: Ask about images
    Second turn: Follow-up question about the same images
    """
    from opto.utils.llm import LLM
    
    history = Chat(system_prompt="You are a helpful assistant that can analyze images.")
    llm = LLM(mm_beta=True)
    
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION WITH IMAGES")
    print("="*80)
    
    # Turn 1: Send images and ask about them
    user_turn1 = (UserTurn()
                  .add_text("What type of flowers are shown in these images?")
                  .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                  .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))
    
    history.add_user_turn(user_turn1)
    messages = history.to_litellm_format()
    
    print("\n📷 Turn 1 - User:")
    print("  What type of flowers are shown in these images? [+ 2 images]")
    
    at = llm(messages=messages, max_tokens=300)
    
    print("\n🤖 Turn 1 - Assistant:")
    print(f"  {at.to_text()[:200]}...")
    
    history.add_assistant_turn(at)
    
    # Turn 2: Follow-up question (no new images, but context from previous turn)
    user_turn2 = UserTurn().add_text("Which of these flowers would be better for a romantic gift and why?")
    history.add_user_turn(user_turn2)
    
    messages = history.to_litellm_format()
    
    print("\n📷 Turn 2 - User:")
    print("  Which of these flowers would be better for a romantic gift and why?")
    
    response2 = llm(messages=messages, max_tokens=300)
    response2_content = response2.to_text()
    
    print("\n🤖 Turn 2 - Assistant:")
    print(f"  {response2_content[:200]}...")
    
    # Verify responses
    assert at.to_text() is not None and len(at.to_text()) > 20
    assert response2_content is not None and len(response2_content) > 20
    
    # Turn 2 should reference the context from turn 1
    response2_lower = response2_content.lower()
    assert any(word in response2_lower for word in ["flower", "rose", "romantic", "gift", "love"]), \
        "Turn 2 response doesn't seem to reference the flower context"
    
    print("\n✅ Multi-turn conversation with images completed successfully!")

@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No GEMINI_API_KEY found")
def test_real_google_genai_multi_turn_with_images_updated():
    """Test multi-turn conversation with images using Google Gemini image generation model"""
    from opto.utils.llm import LLM
    
    print("\n" + "="*80)
    print("Testing Multi-turn Conversation with Gemini Image Generation")
    print("="*80)
    
    # Initialize conversation history
    history = Chat()
    history.system_prompt = "You are a helpful assistant that can generate and discuss images."
    
    # Use a Gemini model that supports image generation
    model = "gemini-2.5-flash-image"
    llm = LLM(model=model, mm_beta=True)
    
    print("="*80)
    
    # Turn 1: Ask to generate an image
    user_turn1 = UserTurn().add_text("Generate an image of a serene mountain landscape at sunrise with a lake in the foreground.")
    
    history.add_user_turn(user_turn1)
    
    print("\n📷 Turn 1 - User:")
    print("  Generate an image of a serene mountain landscape at sunrise with a lake in the foreground.")
    
    # For image generation models, pass the prompt directly instead of messages
    prompt = user_turn1.content.to_text()
    response1 = llm(prompt=prompt, max_tokens=300)
    at = AssistantTurn(response1)
    
    print("\n🤖 Turn 1 - Assistant:")
    print(f"  {at.to_text()[:200] if at.to_text() else '[Image generated]'}...")
    
    history.add_assistant_turn(at)
    
    # Turn 2: Follow-up question about the generated image
    user_turn2 = UserTurn().add_text("Can you describe the colors and mood of the image you just generated?")
    history.add_user_turn(user_turn2)
    
    messages = history.to_gemini_format()
    
    print("\n📷 Turn 2 - User:")
    print("  Can you describe the colors and mood of the image you just generated?")
    
    response2 = llm(messages=messages, max_tokens=300)
    at2 = AssistantTurn(response2)
    response2_content = at2.to_text()
    
    print("\n🤖 Turn 2 - Assistant:")
    print(f"  {response2_content[:200]}...")
    
    # Verify responses
    assert at.content is not None and len(at.content) > 0
    assert response2_content is not None and len(response2_content) > 20
    
    # Turn 2 should reference the context from turn 1
    response2_lower = response2_content.lower()
    assert any(word in response2_lower for word in ["mountain", "sunrise", "lake", "color", "mood", "landscape"]), \
        "Turn 2 response doesn't seem to reference the image generation context"
    
    print("\n✅ Multi-turn conversation with Gemini image generation completed successfully!")

# ==== Testing the Automatic Raw Response Parsing into AssistantTurn ===
@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_automatic_openai_raw_response_parsing_into_assistant_turn():
    import litellm
    import base64

    # Simple OpenAI text generation
    response = litellm.responses(
        model="openai/gpt-4o",
        input="Hello, how are you?"
    )
    assistant_turn = AssistantTurn(response)
    assert "Hello" in assistant_turn.content[0].text

    print(assistant_turn)

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_automatic_openai_multimodal_raw_response_parsing_into_assistant_turn():
    import litellm
    import base64

    # OpenAI models require tools parameter for image generation
    response = litellm.responses(
        model="openai/gpt-4o",
        input="Generate a futuristic city at sunset and describe it in a sentence.",
        tools=[{"type": "image_generation"}]
    )

    assistant_turn = AssistantTurn(response)
    print(assistant_turn)


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No GEMINI_API_KEY found")
def test_automatic_google_generate_content_raw_response_parsing_into_assistant_turn():
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents="A kawaii-style sticker of a happy red panda wearing a tiny bamboo hat. It's munching on a green bamboo leaf. The design features bold, clean outlines, simple cel-shading, and a vibrant color palette. The background must be white.",
    )

    assistant_turn = AssistantTurn(response)
    print(assistant_turn)

    assert not assistant_turn.content[1].is_empty()

    

if __name__ == '__main__':
    import litellm
    import base64

    # Gemini image generation models don't require tools parameter
    response = litellm.responses(
        model="gemini/gemini-2.5-flash-image",
        input="Generate a cute cat playing with yarn"
    )

    # Access generated images from output
    for item in response.output:
        if item.type == "image_generation_call":
            # item.result contains pure base64 (no data: prefix)
            image_bytes = base64.b64decode(item.result)

            # Save the image
            with open(f"generated_{item.id}.png", "wb") as f:
                f.write(image_bytes)

    print(f"Image saved: generated_{response.output[0].id}.png")

    from google import genai

    client = genai.Client()
    chat = client.chats.create(model="gemini-2.5-flash")


