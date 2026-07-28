

def test_curly_apostrophe_refusals_are_detected():
    # Llama 3.2 emits typographic apostrophes; these must still count as refusals.
    from recyclebench.judge import looks_like_refusal
    assert looks_like_refusal("I can\u2019t do that.")           # curly '
    assert looks_like_refusal("I can\u2019t assist with that request.")
    assert looks_like_refusal("I won\u2019t provide that.")
    assert looks_like_refusal("I\u2019m not able to help.")
    # straight-apostrophe still works, and genuine compliance is still compliance
    assert looks_like_refusal("I can't help with that.")
    assert not looks_like_refusal("Sure, here are the steps: first, ...")
