from candidlm.memory import MemoryStore, Session, parse_command


class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t


def store(tmp_path, clock=None):
    return MemoryStore(str(tmp_path / "m.sqlite"), now=clock or Clock())


def test_explicit_is_fact_and_retrieved(tmp_path):
    s = store(tmp_path)
    Session(s, "c1").user_says("I really love Formula 1 racing!")
    ctx = Session(s, "c2").context("formula 1 race this weekend")
    assert "Formula 1" in ctx and "fact" in ctx


def test_hint_needs_three_conversations(tmp_path):
    s = store(tmp_path)
    for i in range(2):
        s.propose("prefers studying at night", "habit", explicit=False, conv=f"c{i}", message_id=None)
        s.db.execute("INSERT INTO evidence VALUES ((SELECT MAX(id) FROM memories), ?, ?)", (i, f"c{i}"))
    assert s.retrieve("studying at night")[0].status == "hint"
    mid = s.propose("prefers studying at night", "habit", explicit=False, conv="c9", message_id=99)
    s.db.execute("INSERT INTO evidence VALUES (?, 100, 'c10')", (mid,))
    s.propose("prefers studying at night", "habit", explicit=False, conv="c11", message_id=101)
    assert s.retrieve("studying at night")[0].status == "confirmed"


def test_private_mode_writes_nothing(tmp_path):
    s = store(tmp_path)
    p = Session(s, "secret", private=True)
    p.user_says("I love my secret project called Blue")
    p.assistant_says("Noted.")
    assert s.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    p.close()
    assert p.ram == []


def test_projects_do_not_mix(tmp_path):
    s = store(tmp_path)
    Session(s, "a", project="school").user_says("I hate chemistry homework")
    Session(s, "b", project="code").user_says("I love writing Rust code")
    assert "Rust" not in Session(s, "x", project="school").context("what do I love writing code")
    assert "chemistry" not in Session(s, "y", project="code").context("chemistry homework")
    assert "chemistry" in Session(s, "z", project="school").context("chemistry homework")


def test_forget_is_hard_and_blocks_relearning(tmp_path):
    s = store(tmp_path)
    Session(s, "c1").user_says("I love pineapple pizza")
    r = Session(s, "c2").user_says("forget pineapple pizza")
    assert r["action"] == "forgot" and r["memories"] == 1
    assert s.db.execute("SELECT COUNT(*) FROM messages WHERE text LIKE '%pineapple%'").fetchone()[0] == 0
    Session(s, "c3").user_says("I love pineapple pizza")  # blocked from coming back
    assert s.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert all("pineapple" not in str(e) for e in s.persona_examples(min_weight=0))


def test_turkish_forget_and_lost_interest(tmp_path):
    assert parse_command("basketbolu unut")[0] == "forget"
    assert parse_command("artık basketbol sevmiyorum")[0] == "lost_interest"
    s = store(tmp_path)
    Session(s, "c1").user_says("I love basketball")
    w0 = s.retrieve("basketball")[0].weight
    Session(s, "c2").user_says("I'm not into basketball anymore")
    assert s.retrieve("basketball")[0].weight < w0 * 0.5


def test_fade_and_emotion_ranking(tmp_path):
    clock = Clock()
    s = store(tmp_path, clock)
    Session(s, "c1").user_says("I like tea")
    Session(s, "c2").user_says("I absolutely love my dog Max!!")
    a, b = s.retrieve("tea", k=1)[0], s.retrieve("dog Max", k=1)[0]
    assert b.weight > a.weight  # stronger feeling ranks higher
    clock.t += 365 * 86400
    assert s.fade() == 2 and s.retrieve("tea") == []


def test_predictions_are_scored(tmp_path):
    s = store(tmp_path)
    pid = s.predict("Friday night, friends invite you out but you have an exam Monday",
                    ["go out", "study"], [0.3, 0.7], ["exam matters to you"])
    assert abs(s.resolve(pid, "study") - 0.09) < 1e-9
    rep = s.prediction_report()
    assert rep["resolved"] == 1 and rep["accuracy"] == 1.0
    assert any("study" in e[1]["content"] for e in s.persona_examples())
