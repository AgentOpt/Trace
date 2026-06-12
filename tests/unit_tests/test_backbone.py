"""Tests for the minimal multimodal backbone.

Covers:
1. Content blocks (TextContent, ImageContent, ContentBlockList, Content)
2. UserTurn / AssistantTurn building and LiteLLM-format rendering
3. The stateless ``to_messages`` helper (system + history + user content)
4. Opt-in live LLM calls (set RUN_LIVE_LLM_TESTS=1) and raw-response parsing
"""
import os
import base64
import pytest

from opto.utils.backbone import (
    TextContent,
    ImageContent,
    ContentBlockList,
    Content,
    UserTurn,
    AssistantTurn,
    to_messages,
    DEFAULT_IMAGE_PLACEHOLDER,
)

# Live tests make real LLM calls. They are opt-in: set RUN_LIVE_LLM_TESTS=1.
SKIP_REASON = "Live LLM test; set RUN_LIVE_LLM_TESTS=1 to run"
HAS_CREDENTIALS = os.environ.get("RUN_LIVE_LLM_TESTS") == "1"


# ============================================================================
# Content block tests
# ============================================================================

def test_text_content_merge():
    """Consecutive text blocks merge into one."""
    blocks = ContentBlockList()
    blocks.append("Hello")
    blocks.append("world")
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Hello world"


def test_image_content_url_format():
    img = ImageContent(image_url="https://example.com/a.png")
    fmt = img.to_litellm_format()
    assert fmt["type"] == "input_image"
    assert fmt["image_url"] == "https://example.com/a.png"


def test_image_content_base64_format():
    data = base64.b64encode(b"fake image").decode("utf-8")
    img = ImageContent(image_data=data, media_type="image/png")
    fmt = img.to_litellm_format()
    assert fmt["type"] == "input_image"
    assert fmt["image_url"].startswith("data:image/png;base64,")


def test_content_block_list_has_images_and_to_text():
    blocks = ContentBlockList()
    blocks.append("before")
    blocks.append(ImageContent(image_url="https://example.com/a.png"))
    blocks.append("after")
    assert blocks.has_images()
    text = blocks.to_text()
    assert "before" in text and "after" in text
    assert DEFAULT_IMAGE_PLACEHOLDER.strip() in text


def test_content_variadic_builder():
    ctx = Content("some text", "more text")
    assert not ctx.has_images()
    assert "some text" in ctx.to_text()


def test_content_template_builder():
    img = ImageContent(image_url="https://example.com/a.png")
    ctx = Content(f"See {DEFAULT_IMAGE_PLACEHOLDER} here", images=[img])
    assert ctx.has_images()


def test_content_blocks_to_litellm_format_mixed():
    blocks = ContentBlockList()
    blocks.append("text")
    blocks.append(ImageContent(image_url="https://example.com/a.png"))
    fmt = blocks.to_litellm_format(role="user")
    assert len(fmt) == 2
    assert fmt[0]["type"] == "input_text"
    assert fmt[1]["type"] == "input_image"


# ============================================================================
# Turn tests
# ============================================================================

def test_user_turn_multiple_images():
    user_turn = (UserTurn()
                 .add_text("What are in these images?")
                 .add_image(url="https://example.com/one.jpg")
                 .add_image(url="https://example.com/two.jpg"))
    msg = user_turn.to_litellm_format()
    assert msg["role"] == "user"
    assert len(msg["content"]) == 3
    assert msg["content"][0]["type"] == "input_text"
    assert msg["content"][1]["type"] == "input_image"
    assert msg["content"][2]["type"] == "input_image"


def test_user_turn_base64_images():
    d1 = base64.b64encode(b"img1").decode("utf-8")
    d2 = base64.b64encode(b"img2").decode("utf-8")
    user_turn = (UserTurn()
                 .add_text("Compare:")
                 .add_image(data=d1, media_type="image/png")
                 .add_image(data=d2, media_type="image/jpeg"))
    msg = user_turn.to_litellm_format()
    assert msg["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert msg["content"][2]["image_url"].startswith("data:image/jpeg;base64,")


def test_assistant_turn_text_format():
    at = AssistantTurn().add_text("Here is the answer.")
    msg = at.to_litellm_format()
    assert msg["role"] == "assistant"
    assert any("answer" in item.get("text", "") for item in msg["content"])


def test_assistant_turn_get_text_and_images():
    at = (AssistantTurn()
          .add_text("caption")
          .add_image(url="https://example.com/gen.png"))
    assert "caption" in at.to_text()
    assert at.get_images().has_images()
    assert len(at.get_images()) == 1
    assert at.get_text().to_text().strip() == "caption"


# ============================================================================
# to_messages helper tests
# ============================================================================

def test_to_messages_system_and_user():
    messages = to_messages("You are helpful.", "Hello")
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "Hello"


def test_to_messages_no_system():
    messages = to_messages(None, "Hi")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_to_messages_with_history():
    history = [
        UserTurn().add_text("Q1").to_litellm_format(),
        AssistantTurn().add_text("A1").to_litellm_format(),
    ]
    messages = to_messages("sys", "Q2", history=history)
    # system + 2 history + current user
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"][0]["text"] == "Q2"


def test_to_messages_multimodal_user_content():
    blocks = ContentBlockList()
    blocks.append("Analyze")
    blocks.append(ImageContent(image_url="https://example.com/a.png"))
    messages = to_messages("sys", blocks)
    user_content = messages[-1]["content"]
    assert len(user_content) == 2
    assert user_content[1]["type"] == "input_image"


# ============================================================================
# Live LLM call tests (opt-in)
# ============================================================================

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_real_llm_call_with_multiple_images():
    from opto.utils.llm import LLM

    user_turn = (UserTurn()
                 .add_text("What are in these images? Describe each briefly.")
                 .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                 .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))

    messages = to_messages("You analyze images.", user_turn.content)

    llm = LLM(mm_beta=True)
    response = llm(messages=messages, max_tokens=500)
    text = response.to_text()

    assert text is not None and len(text) > 50
    assert any(w in text.lower() for w in ["flower", "image", "rose", "pink", "red", "petal"])


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_real_llm_multi_turn_with_images():
    from opto.utils.llm import LLM

    llm = LLM(mm_beta=True)
    history = []

    user1 = (UserTurn()
             .add_text("What type of flowers are shown in these images?")
             .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg"))
    user1_msg = user1.to_litellm_format()
    messages = to_messages("You analyze images.", history=history)
    messages.append(user1_msg)

    response1 = llm(messages=messages, max_tokens=300)
    history.append(user1_msg)
    history.append(response1.to_litellm_format())

    user2_msg = UserTurn().add_text("Which would be better for a gift and why?").to_litellm_format()
    messages = to_messages("You analyze images.", history=history)
    messages.append(user2_msg)

    response2 = llm(messages=messages, max_tokens=300)
    text2 = response2.to_text()

    assert response1.to_text() and len(response1.to_text()) > 20
    assert text2 and len(text2) > 20
    assert any(w in text2.lower() for w in ["flower", "rose", "gift", "love"])


# ============================================================================
# Raw-response parsing into AssistantTurn (opt-in)
# ============================================================================

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_openai_raw_response_parsing():
    import litellm

    response = litellm.responses(model="openai/gpt-4o", input="Hello, how are you?")
    at = AssistantTurn(response)
    assert "Hello" in at.content[0].text or len(at.to_text()) > 0


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No GEMINI_API_KEY found")
def test_google_generate_content_parsing():
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents="A kawaii sticker of a happy red panda. White background.",
    )
    at = AssistantTurn(response)
    assert len(at.content) > 0
