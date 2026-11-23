"""CandidLM memory: knows-me mode, private mode, project memory, forgetting, predictions.

Everything lives in one local SQLite file on the user's PC. Nothing here is ever uploaded.

Rules (from the plan):
- Evidence only. A memory keeps links to the messages behind it. Explicit statements are facts
  ("confirmed"); anything implied is a "hint" until it shows up in >= 3 separate conversations.
- Importance = frequency x recency x feeling (x2 if pinned, x0.5 while still a hint).
  Recency halves every `half_life_days` unless the topic comes back.
- Natural fade: below `archive_below` a memory is archived (kept, but not used).
- "Not into that anymore": weight goes down a lot, memory stays (it is still true that it was liked).
- "Forget X": the memory, its evidence and the matching message text are deleted, and X goes on a
  blocklist that also filters persona-training data, so the adapter (retrained from scratch) loses it too.
- Private mode: nothing from that conversation is written anywhere.
- Projects: each memory is tagged with a project (or global). Retrieval inside a project only sees
  that project plus global memories.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

DAY = 86400.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY, conv TEXT, project TEXT, role TEXT, text TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY, project TEXT, kind TEXT, text TEXT, status TEXT,
  freq REAL, last_seen REAL, created REAL, emotion REAL, pinned INTEGER DEFAULT 0,
  interest REAL DEFAULT 1.0, emb BLOB);
CREATE TABLE IF NOT EXISTS evidence(memory_id INTEGER, message_id INTEGER, conv TEXT);
CREATE TABLE IF NOT EXISTS blocklist(id INTEGER PRIMARY KEY, phrase TEXT, project TEXT, emb BLOB, ts REAL);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, project TEXT, situation TEXT, options TEXT, predicted TEXT,
  probs TEXT, reasons TEXT, actual TEXT, ts REAL, resolved_ts REAL);
CREATE INDEX IF NOT EXISTS mem_project ON memories(project);
"""


# --------------------------------------------------------------------------- embeddings


def hash_embed(text: str, dim: int = 512) -> np.ndarray:
    """Offline fallback embedding: hashed word + bigram counts. Swap in a real embedding model on the PC."""
    words = re.findall(r"\w+", text.lower())
    v = np.zeros(dim, dtype=np.float32)
    feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
    for w in feats:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------------------------------------------------------- feeling + statement detection

EMOTION_WORDS = {
    # English
    "love": 1.0, "adore": 1.0, "obsessed": 0.9, "hate": 0.9, "excited": 0.8, "amazing": 0.7, "angry": 0.8,
    "furious": 0.9, "favorite": 0.7, "favourite": 0.7, "really": 0.2, "so": 0.1, "scared": 0.7, "sad": 0.6,
    # Turkish
    "seviyorum": 0.9, "bayılıyorum": 1.0, "nefret": 0.9, "aşığım": 1.0, "heyecanlı": 0.8, "kızgın": 0.8,
    "harika": 0.7, "çok": 0.2, "favori": 0.7, "korkuyorum": 0.7, "üzgün": 0.6,
}


def emotion_score(text: str) -> float:
    t = text.lower()
    s = sum(w for k, w in EMOTION_WORDS.items() if re.search(rf"\b{re.escape(k)}\b", t))
    s += 0.1 * min(3, t.count("!"))
    return float(min(1.0, s))


FORGET = [r"^\s*(?:please\s+)?forget(?: about| that)?\s+(?P<x>.+?)\s*[.!]*$",
          r"^\s*(?P<x>.+?)\s*(?:'?(?:y[iıuü]|[iıuü]))?\s+unut\b.*$"]
LOST_INTEREST = [r"i(?:'m| am) not (?:into|interested in) (?P<x>.+?) anymore",
                 r"i don'?t (?:like|care about) (?P<x>.+?) anymore",
                 r"artık (?P<x>.+?) (?:sevmiyorum|ilgilenmiyorum|umrumda değil)"]
EXPLICIT = [  # (pattern, kind)
    (r"\bmy name is (?P<x>[^.,!]+)", "identity"),
    (r"\bi (?:really |\w+ly )?(?:love|like|adore|enjoy) (?P<x>[^.!?]+)", "likes"),
    (r"\bi (?:really |\w+ly )?(?:hate|dislike|can'?t stand) (?P<x>[^.!?]+)", "dislikes"),
    (r"\bi(?:'m| am) (?:a |an )?(?P<x>(?:student|developer|engineer|from|learning|studying)[^.!?]*)", "identity"),
    (r"\bbenim adım (?P<x>[^.,!]+)", "identity"),
    (r"(?P<x>[^.!?]+?) (?:çok )?(?:seviyorum|bayılıyorum)", "likes"),
    (r"(?P<x>[^.!?]+?) nefret ediyorum", "dislikes"),
]


def parse_command(text: str):
    t = text.strip()
    for p in FORGET:
        m = re.match(p, t, re.I)
        if m:
            return "forget", m.group("x").strip()
    for p in LOST_INTEREST:
        m = re.search(p, t, re.I)
        if m:
            return "lost_interest", m.group("x").strip()
    return None, None


def extract_explicit(text: str):
    out = []
    for p, kind in EXPLICIT:
        for m in re.finditer(p, text, re.I):
            x = m.group("x").strip(" .,!?")
            if 1 <= len(x) <= 120:
                out.append((kind, x))
    return out


# --------------------------------------------------------------------------- store


@dataclass
class Memory:
    id: int
    project: str | None
    kind: str
    text: str
    status: str
    weight: float
    confidence: float
    evidence: int
    conversations: int


class MemoryStore:
    def __init__(self, path: str = "candidlm_memory.sqlite", embed=hash_embed, half_life_days: float = 30.0,
                 archive_below: float = 0.05, merge_threshold: float = 0.82, now=time.time):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.embed, self.half_life, self.archive_below = embed, half_life_days, archive_below
        self.merge_threshold, self.now = merge_threshold, now

    # ---- scoring
    def weight(self, row) -> float:
        freq, last_seen, emotion, pinned, interest, status = row
        age = max(0.0, (self.now() - last_seen) / DAY)
        w = math.log1p(freq) * 0.5 ** (age / self.half_life) * (1.0 + emotion) * interest
        if pinned:
            w *= 2.0
        if status == "hint":
            w *= 0.5
        return w

    def _blocked(self, text: str, project: str | None) -> bool:
        e = self.embed(text)
        for phrase, proj, emb in self.db.execute("SELECT phrase, project, emb FROM blocklist"):
            if proj is not None and proj != project:
                continue
            if phrase.lower() in text.lower() or cos(e, np.frombuffer(emb, dtype=np.float32)) > self.merge_threshold:
                return True
        return False

    # ---- write path (never called in private mode)
    def add_message(self, conv: str, role: str, text: str, project: str | None = None) -> int:
        cur = self.db.execute("INSERT INTO messages(conv, project, role, text, ts) VALUES (?,?,?,?,?)",
                              (conv, project, role, text, self.now()))
        self.db.commit()
        return cur.lastrowid

    def propose(self, text: str, kind: str, *, explicit: bool, conv: str, message_id: int | None,
                project: str | None = None, emotion: float | None = None) -> int | None:
        """Add evidence for a memory. Merges with a similar existing memory in the same project."""
        if self._blocked(text, project):
            return None
        emo = emotion_score(text) if emotion is None else emotion
        e = self.embed(text)
        best, best_sim = None, 0.0
        for mid, emb in self.db.execute(
                "SELECT id, emb FROM memories WHERE project IS ? AND kind = ? AND status != 'forgotten'",
                (project, kind)):
            s = cos(e, np.frombuffer(emb, dtype=np.float32))
            if s > best_sim:
                best, best_sim = mid, s
        now = self.now()
        if best is not None and best_sim >= self.merge_threshold:
            self.db.execute("UPDATE memories SET freq = freq + 1, last_seen = ?, emotion = MAX(emotion, ?), "
                            "status = CASE WHEN status = 'archived' THEN 'hint' ELSE status END WHERE id = ?",
                            (now, emo, best))
            mid = best
        else:
            cur = self.db.execute(
                "INSERT INTO memories(project, kind, text, status, freq, last_seen, created, emotion, emb) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (project, kind, text, "confirmed" if explicit else "hint", 1.0, now, now, emo, e.tobytes()))
            mid = cur.lastrowid
        if message_id is not None:
            self.db.execute("INSERT INTO evidence(memory_id, message_id, conv) VALUES (?,?,?)", (mid, message_id, conv))
        if explicit:
            self.db.execute("UPDATE memories SET status = 'confirmed' WHERE id = ?", (mid,))
        else:  # promote a hint once it appears in 3+ different conversations
            n_conv = self.db.execute("SELECT COUNT(DISTINCT conv) FROM evidence WHERE memory_id = ?", (mid,)).fetchone()[0]
            if n_conv >= 3:
                self.db.execute("UPDATE memories SET status = 'confirmed' WHERE id = ? AND status = 'hint'", (mid,))
        self.db.commit()
        return mid

    def confirm(self, memory_id: int) -> None:
        self.db.execute("UPDATE memories SET status = 'confirmed' WHERE id = ?", (memory_id,))
        self.db.commit()

    def pin(self, memory_id: int, on: bool = True) -> None:
        self.db.execute("UPDATE memories SET pinned = ? WHERE id = ?", (int(on), memory_id))
        self.db.commit()

    def lost_interest(self, phrase: str, project: str | None = None) -> int:
        ids = self._match(phrase, project)
        for i in ids:
            self.db.execute("UPDATE memories SET interest = interest * 0.2, emotion = 0 WHERE id = ?", (i,))
        self.db.commit()
        return len(ids)

    def forget(self, phrase: str, project: str | None = None) -> dict:
        """Hard delete: memories, their evidence, and message text that mentions the phrase."""
        ids = self._match(phrase, project)
        for i in ids:
            self.db.execute("DELETE FROM evidence WHERE memory_id = ?", (i,))
            self.db.execute("DELETE FROM memories WHERE id = ?", (i,))
        q = "DELETE FROM messages WHERE LOWER(text) LIKE ?" + ("" if project is None else " AND project IS ?")
        args = [f"%{phrase.lower()}%"] + ([] if project is None else [project])
        n_msg = self.db.execute(q, args).rowcount
        self.db.execute("INSERT INTO blocklist(phrase, project, emb, ts) VALUES (?,?,?,?)",
                        (phrase, project, self.embed(phrase).tobytes(), self.now()))
        self.db.commit()
        return {"memories": len(ids), "messages": n_msg}

    def delete_project(self, project: str) -> None:
        mids = [r[0] for r in self.db.execute("SELECT id FROM memories WHERE project = ?", (project,))]
        for i in mids:
            self.db.execute("DELETE FROM evidence WHERE memory_id = ?", (i,))
        self.db.execute("DELETE FROM memories WHERE project = ?", (project,))
        self.db.execute("DELETE FROM messages WHERE project = ?", (project,))
        self.db.commit()

    def _match(self, phrase: str, project: str | None) -> list[int]:
        e = self.embed(phrase)
        out = []
        q = "SELECT id, text, emb FROM memories" + ("" if project is None else " WHERE project IS ?")
        for mid, text, emb in self.db.execute(q, () if project is None else (project,)):
            if phrase.lower() in text.lower() or cos(e, np.frombuffer(emb, dtype=np.float32)) > 0.6:
                out.append(mid)
        return out

    # ---- maintenance
    def fade(self) -> int:
        n = 0
        for mid, *row in self.db.execute(
                "SELECT id, freq, last_seen, emotion, pinned, interest, status FROM memories "
                "WHERE status IN ('hint','confirmed')").fetchall():
            if self.weight(row) < self.archive_below:
                self.db.execute("UPDATE memories SET status = 'archived' WHERE id = ?", (mid,))
                n += 1
        self.db.commit()
        return n

    # ---- read path (allowed in private mode, if the setting says so)
    def retrieve(self, query: str, project: str | None = None, k: int = 8, min_sim: float = 0.05) -> list[Memory]:
        e = self.embed(query)
        rows = self.db.execute(
            "SELECT id, project, kind, text, status, freq, last_seen, emotion, pinned, interest, emb FROM memories "
            "WHERE status IN ('hint','confirmed') AND (project IS NULL OR project IS ?)", (project,)).fetchall()
        scored = []
        for (mid, proj, kind, text, status, freq, last_seen, emo, pinned, interest, emb) in rows:
            sim = cos(e, np.frombuffer(emb, dtype=np.float32))
            if sim < min_sim and not pinned:
                continue
            w = self.weight((freq, last_seen, emo, pinned, interest, status))
            ev, nconv = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT conv) FROM evidence WHERE memory_id = ?",
                                        (mid,)).fetchone()
            conf = 0.95 if status == "confirmed" else min(0.6, 0.2 + 0.15 * nconv)
            scored.append((sim * w, Memory(mid, proj, kind, text, status, w, conf, ev, nconv)))
        scored.sort(key=lambda t: -t[0])
        return [m for _, m in scored[:k]]

    def context_block(self, query: str, project: str | None = None, k: int = 8) -> str:
        """Text placed in the system prompt before a reply."""
        mems = self.retrieve(query, project, k)
        if not mems:
            return ""
        lines = ["Known about the user (use only if relevant; hints are unconfirmed, ask before relying on them):"]
        for m in mems:
            tag = "fact" if m.status == "confirmed" else "hint"
            lines.append(f"- [{tag}, {m.confidence:.0%}] ({m.kind}) {m.text}")
        return "\n".join(lines)

    # ---- predictions of what the user will do
    def predict(self, situation: str, options: list[str], probs: list[float], reasons: list[str],
                project: str | None = None) -> int:
        top = options[int(np.argmax(probs))]
        cur = self.db.execute("INSERT INTO predictions(project, situation, options, predicted, probs, reasons, ts) "
                              "VALUES (?,?,?,?,?,?,?)", (project, situation, json.dumps(options), top,
                                                         json.dumps(probs), json.dumps(reasons), self.now()))
        self.db.commit()
        return cur.lastrowid

    def resolve(self, prediction_id: int, actual: str) -> float:
        opts, probs = self.db.execute("SELECT options, probs FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        opts, probs = json.loads(opts), json.loads(probs)
        brier = sum((p - (1.0 if o == actual else 0.0)) ** 2 for o, p in zip(opts, probs)) / len(opts)
        self.db.execute("UPDATE predictions SET actual = ?, resolved_ts = ? WHERE id = ?",
                        (actual, self.now(), prediction_id))
        self.db.commit()
        return brier

    def prediction_report(self) -> dict:
        rows = self.db.execute("SELECT options, probs, predicted, actual FROM predictions WHERE actual IS NOT NULL").fetchall()
        if not rows:
            return {"resolved": 0}
        acc, brier = 0, 0.0
        for o, p, pred, act in rows:
            o, p = json.loads(o), json.loads(p)
            acc += pred == act
            brier += sum((pp - (1.0 if oo == act else 0.0)) ** 2 for oo, pp in zip(o, p)) / len(o)
        return {"resolved": len(rows), "accuracy": acc / len(rows), "brier": brier / len(rows)}

    # ---- persona adapter training data (blocklist applied)
    def persona_examples(self, min_weight: float = 0.3, name: str = "Deniz") -> list[dict]:
        """Conversations for the persona LoRA: the user's past decisions + stable memories.
        Always formal answers; forgotten/blocked content is filtered out."""
        ex = []
        for situation, options, actual in self.db.execute(
                "SELECT situation, options, actual FROM predictions WHERE actual IS NOT NULL"):
            if self._blocked(situation + " " + actual, None):
                continue
            ex.append([{"role": "user", "content": f"In this situation, what would {name} most likely do? "
                                                   f"{situation} Options: {', '.join(json.loads(options))}"},
                       {"role": "assistant", "content": f"{name} would most likely choose: {actual}."}])
        for mid, proj, kind, text, *row in self.db.execute(
                "SELECT id, project, kind, text, freq, last_seen, emotion, pinned, interest, status FROM memories "
                "WHERE status = 'confirmed'").fetchall():
            if self.weight(tuple(row)) < min_weight or self._blocked(text, proj):
                continue
            ex.append([{"role": "user", "content": f"What do you know about {name} regarding {kind}?"},
                       {"role": "assistant", "content": f"Based on what {name} has said, {kind}: {text}."}])
        return ex


class Session:
    """One conversation. Handles commands, extraction, and private mode."""

    def __init__(self, store: MemoryStore, conv_id: str, project: str | None = None, private: bool = False,
                 knows_me: bool = True, read_in_private: bool = True):
        self.store, self.conv, self.project = store, conv_id, project
        self.private, self.knows_me, self.read_in_private = private, knows_me, read_in_private
        self.ram: list[tuple[str, str]] = []  # private-mode transcript, never written to disk

    def user_says(self, text: str) -> dict:
        """Call for every user message. Returns what happened (for the UI)."""
        cmd, x = parse_command(text)
        if cmd == "forget":
            return {"action": "forgot", **self.store.forget(x, None)}  # forgetting is global and immediate
        if self.private:
            self.ram.append(("user", text))
            if cmd == "lost_interest":
                return {"action": "noted_in_private_only"}
            return {"action": "none"}
        mid = self.store.add_message(self.conv, "user", text, self.project)
        if cmd == "lost_interest":
            return {"action": "lowered", "memories": self.store.lost_interest(x, None)}
        if not self.knows_me:
            return {"action": "none"}
        emo = emotion_score(text)  # feeling is read from the whole message, not just the extracted phrase
        added = [self.store.propose(t, kind, explicit=True, conv=self.conv, message_id=mid, project=self.project,
                                    emotion=emo) for kind, t in extract_explicit(text)]
        return {"action": "learned" if any(added) else "none", "memories": [a for a in added if a]}

    def assistant_says(self, text: str) -> None:
        if self.private:
            self.ram.append(("assistant", text))
        else:
            self.store.add_message(self.conv, "assistant", text, self.project)

    def context(self, query: str) -> str:
        if not self.knows_me or (self.private and not self.read_in_private):
            return ""
        return self.store.context_block(query, self.project)

    def close(self) -> None:
        self.ram.clear()
