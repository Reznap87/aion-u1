import pytest

from aion_u1.training.evaluate import encode_chat_prompt


class Tokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "test question"}]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "rendered"

    def __call__(self, prompt, **kwargs):
        assert prompt == "rendered"
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        return {"input_ids": [1, 2]}


def test_instruct_prompt_format():
    assert encode_chat_prompt(Tokenizer(), "test question") == {"input_ids": [1, 2]}


def test_missing_chat_template_rejected():
    tokenizer = Tokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        encode_chat_prompt(tokenizer, "test question")
