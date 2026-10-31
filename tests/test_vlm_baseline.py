from s1geo.vlm_baseline import build_prompt, parse_answer

Q = {"type": "choice", "instructions": "Which?", "criteria": {"forest": "Forest.", "arable_land": "Arable land."}}


def test_prompt_and_letter_parsing():
    prompt, keys, letters = build_prompt(Q)
    assert "A. Forest" in prompt and "B. Arable land" in prompt and letters == ["A", "B"]
    assert parse_answer("B", Q, keys, letters) == "arable_land"
    assert parse_answer("A. Forest", Q, keys, letters) == "forest"
    assert parse_answer("The answer is arable land.", Q, keys, letters) == "arable_land"
    assert parse_answer("I am not sure", Q, keys, letters) is None


def test_noul_parsing():
    q = {"type": "noul", "instructions": "Is there water?"}
    _, keys, letters = build_prompt(q)
    assert parse_answer("Yes, there is.", q, keys, letters) == "yes"
    assert parse_answer("no", q, keys, letters) == "no"
    assert parse_answer("Maybe", q, keys, letters) is None


def test_reasoning_style_outputs():
    _, keys, letters = build_prompt(Q)
    assert parse_answer("Looking at the image, fields dominate.\n\nThe answer is B.", Q, keys, letters) == "arable_land"
    assert parse_answer("Some reasoning...\n\n**Answer: A**", Q, keys, letters) == "forest"
    assert parse_answer("Reasoning about a field.\nB\n", Q, keys, letters) == "arable_land"


def test_bold_final_answer():
    _, keys, letters = build_prompt(Q)
    text = "Forest is visible at the edge but the dominant class is **arable land**, the fields."
    assert parse_answer(text, Q, keys, letters) == "arable_land"


def test_open_lc_parsing():
    q = {"type": "open_lc", "instructions": "What L2 land cover classes are there in the image?"}
    assert parse_answer("Forests, marine waters", q, [], []) == "forests|marine waters"
    assert parse_answer("None", q, [], []) == ""
    assert parse_answer("Lots of trees and a lake", q, [], []) is None


def test_inline_options():
    q = {"type": "choice", "instructions": "How many? a) 0, b) 2, c) 5, d) 3", "criteria": {"a": "0", "b": "2", "c": "5", "d": "3"},
         "inline_options": True}
    prompt, keys, letters = build_prompt(q)
    assert prompt.startswith("How many? a) 0") and letters == ["A", "B", "C", "D"]
    assert parse_answer("c)", q, keys, letters) == "c"
    assert parse_answer("B", q, keys, letters) == "b"
    assert parse_answer("The answer is d.", q, keys, letters) == "d"
