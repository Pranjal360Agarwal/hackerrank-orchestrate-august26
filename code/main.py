#!/usr/bin/env python3
"""Deterministic, context-aware WhatsApp notification router.

Run from the repository root with ``python code/main.py``.  The router only
uses participant-facing files below ``dataset`` and writes ``output.csv`` in
the repository root by default.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from media import MediaAnalyzer


OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "can", "for",
    "from", "has", "have", "i", "if", "in", "is", "it", "just", "of",
    "on", "or", "our", "please", "the", "this", "to", "we", "with", "you",
    "your", "now", "today", "will", "when", "all", "my", "not", "do",
}
RISK_PATTERNS = (
    r"\b(?:share|send|reply with|enter|confirm|verify).{0,50}\b(?:otp|one[- ]?time|password|pin|login code|verification code)\b",
    r"\b(?:otp|password|pin|login code|verification code).{0,50}\b(?:share|send|reply|enter|confirm|verify)\b",
    r"\b(?:account|profile|workspace|wallet).{0,35}\b(?:blocked|suspend|expire|disabled).{0,50}\b(?:otp|password|code|verify|confirm)\b",
    r"\bignore (?:all |previous )?(?:instructions|rules)\b",
)
URGENT_WORDS = {
    "urgent", "emergency", "immediately", "asap", "right now", "come online",
    "call me", "call?", "deadline", "escalation", "blocked", "expires today",
    "last-minute", "last minute", "minutes", "today", "tonight", "eod",
}
EVENT_WORDS = {
    "meeting", "bus", "school", "pickup", "plumber", "water", "tanker",
    "maintenance", "circular", "consent", "appointment", "event", "practice",
    "timing", "schedule", "leaving", "form", "registrations",
}
PROMO_WORDS = {
    "offer", "sale", "discount", "coupon", "promo", "shop", "buy", "order now",
    "save", "% off", "deal", "cashback", "selected products", "new here",
    "selling", "for sale", "unsubscribe", "marketing messages", "itinerary",
}
GREETING_WORDS = {"good morning", "good night", "good evening", "blessings", "good vibes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]{2,}", text.lower()) if word not in STOP_WORDS}


def phrase_present(text: str, phrases: Iterable[str]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Evidence:
    message_id: str
    score: float
    opened: int
    replied: int
    dismissed: int
    muted: int
    reported: int


class Router:
    def __init__(self, dataset: Path) -> None:
        self.dataset = dataset
        self.users = self._by_key(read_csv(dataset / "users.csv"), "user_id")
        self.groups = self._by_key(read_csv(dataset / "groups.csv"), "group_id")
        self.businesses = self._by_key(read_csv(dataset / "business_accounts.csv"), "business_id")
        self.group_members = {
            (row["group_id"], row["user_id"]): row
            for row in read_csv(dataset / "group_members.csv")
        }
        self.business_history = {
            (row["user_id"], row["business_id"]): row
            for row in read_csv(dataset / "user_business_history.csv")
        }
        self.events = {
            row["message_id"]: row for row in read_csv(dataset / "message_events.csv")
        }
        self.media = MediaAnalyzer(dataset)
        self.history_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(dataset / "message_history.csv"):
            self.history_by_user[row["user_id"]].append(row)

    @staticmethod
    def _by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
        return {row[key]: row for row in rows}

    def classify_type(self, message: dict[str, str]) -> str:
        text = message.get("message_text", "").lower()
        if any(re.search(pattern, text, flags=re.DOTALL) for pattern in RISK_PATTERNS):
            return "scam"
        if "safety advisory" in text or "never ask for otp" in text:
            return "business_update"
        if phrase_present(text, PROMO_WORDS):
            return "promotion"
        if re.search(r"\b(?:rs\.?|₹)\s?\d", text):
            return "promotion"
        if "pickup" in text and any(word in text for word in ("photos", "pics", "attached", "set", "helmet")):
            return "promotion"
        if phrase_present(text, GREETING_WORDS):
            return "greeting"
        if any(word in text for word in ("payment", "invoice", "card", "banking", "transaction", "refund")):
            return "payment"
        if message.get("conversation_type") == "personal" and (
            "found your number" in text or "volunteer sheet" in text
        ):
            return "unknown"
        if (message.get("conversation_type") != "personal") and phrase_present(text, EVENT_WORDS):
            return "event"
        if message.get("forwarded_count") not in ("", "0") or "fwd" in text or "forward" in text:
            return "forward"
        if message.get("conversation_type") == "group" and (
            f"@{message.get('user_id', '').lower()}" in text or "anyone" in text or "dm if" in text
        ):
            return "personal"
        if message.get("conversation_type") == "business":
            return "business_update"
        if message.get("conversation_type") == "personal":
            return "personal"
        return "unknown"

    def evidence_for(self, message: dict[str, str], category: str) -> list[Evidence]:
        current_terms = tokens(message.get("message_text", ""))
        results: list[Evidence] = []
        for prior in self.history_by_user.get(message["user_id"], []):
            prior_terms = tokens(prior.get("message_text", ""))
            overlap = len(current_terms & prior_terms) / max(1, len(current_terms | prior_terms))
            same_source = (
                prior.get("group_id") and prior.get("group_id") == message.get("group_id")
            ) or (
                prior.get("business_id") and prior.get("business_id") == message.get("business_id")
            ) or (
                prior.get("sender_user_id") and prior.get("sender_user_id") == message.get("sender_user_id")
            )
            prior_type = self.classify_type(prior)
            score = overlap + (0.42 if same_source else 0) + (0.16 if prior_type == category else 0)
            if score < 0.20:
                continue
            event = self.events.get(prior["message_id"], {})
            results.append(Evidence(
                message_id=prior["message_id"], score=score,
                opened=as_int(event.get("message_opened")), replied=as_int(event.get("message_replied")),
                dismissed=as_int(event.get("notification_dismissed")), muted=as_int(event.get("muted_after_message")),
                reported=as_int(event.get("message_reported")),
            ))
        return sorted(results, key=lambda item: item.score, reverse=True)[:3]

    def route(self, message: dict[str, str]) -> dict[str, str]:
        text = message.get("message_text", "")
        lower = text.lower()
        category = self.classify_type(message)
        evidence = self.evidence_for(message, category)
        evidence_ids = ";".join(item.message_id for item in evidence) or "none"
        history_positive = sum(item.opened + item.replied for item in evidence)
        history_negative = sum(item.dismissed + item.muted + item.reported for item in evidence)
        strongest_positive = max((item.score for item in evidence if item.opened or item.replied), default=0.0)
        strongest_negative = max(
            (item.score for item in evidence if item.dismissed or item.muted or item.reported), default=0.0
        )
        conversation = message.get("conversation_type", "")
        user = self.users.get(message["user_id"], {})
        urgent = phrase_present(lower, URGENT_WORDS) or bool(re.search(r"\b\d{1,2}\s*(?:min|mins|minutes)\b", lower))
        if "nothing urgent" in lower or "not urgent" in lower:
            urgent = False
        direct_mention = f"@{message['user_id'].lower()}" in lower

        # Safety is deliberately non-negotiable: personalization never overrides it.
        if category == "scam":
            return self._result(message, "mute", "scam", 0.97, evidence_ids,
                                "The message requests credentials or uses a coercive account-security claim.")
        if category == "forward" and (as_int(message.get("forwarded_count")) >= 3 or "forward" in lower):
            return self._result(message, "mute", "forward", 0.87, evidence_ids,
                                "This is a heavily forwarded, low-accountability message that can be suppressed.")
        if history_negative >= max(2, history_positive + 1) and strongest_negative >= strongest_positive:
            return self._result(message, "mute", category, 0.84, evidence_ids,
                                "Similar messages were repeatedly dismissed, muted, or reported by this user.")

        if conversation == "business":
            business = self.businesses.get(message.get("business_id", ""), {})
            relationship = self.business_history.get((message["user_id"], message.get("business_id", "")), {})
            trusted = business.get("verified") == "1" and business.get("official_domain") == business.get("domain_used_by_sender")
            opted_out = relationship.get("allows_promotions") == "0" or bool(relationship.get("promotions_opted_out_at"))
            active = as_int(relationship.get("activity_count_180d")) > 0
            if category == "promotion":
                if opted_out or not active:
                    return self._result(message, "mute", "promotion", 0.86, evidence_ids,
                                        "This promotional sender is not useful for this user or has been opted out.")
                return self._result(message, "digest", "promotion", 0.73, evidence_ids,
                                    "A known business sent an optional promotion that can wait for a digest.")
            if category in {"payment", "business_update", "event"} and trusted and active:
                if category == "event" and any(word in lower for word in ("appointment", "prescription", "pickup", "scheduled time")):
                    return self._result(message, "notify", "event", 0.84, evidence_ids,
                                        "A trusted health or appointment update needs attention before its scheduled time.")
                if urgent or any(word in lower for word in ("delivery", "appointment", "packed", "due today", "failed")):
                    return self._result(message, "notify", "business_update", 0.84, evidence_ids,
                                        "A trusted business sent a time-sensitive update tied to the user's relationship.")
                return self._result(message, "digest", category, 0.76, evidence_ids,
                                    "A trusted business update is useful but does not require an interruption.")
            return self._result(message, "digest", category, 0.62, evidence_ids,
                                "This business message appears legitimate but can be reviewed later.")

        if conversation == "group":
            membership = self.group_members.get((message.get("group_id", ""), message["user_id"]), {})
            group = self.groups.get(message.get("group_id", ""), {})
            muted = membership.get("group_muted_by_user") == "1"
            sender_membership = self.group_members.get(
                (message.get("group_id", ""), message.get("sender_user_id", "")), {}
            )
            sender_is_admin = sender_membership.get("role") == "admin" or "admin" in lower
            operational = category == "event" and (urgent or sender_is_admin)
            critical_operational = operational and any(
                word in lower for word in ("tanker", "water", "valve", "fire", "gas leak", "evacuate", "power outage")
            )
            if muted and not (direct_mention or operational):
                return self._result(message, "mute", category, 0.82, evidence_ids,
                                    "The user muted this group and the message has no direct or urgent exception.")
            if direct_mention and (urgent or "work" in group.get("group_type", "")):
                output_type = "urgent" if "work" in group.get("group_type", "") else "personal"
                return self._result(message, "notify", output_type, 0.88, evidence_ids,
                                    "A direct mention creates an immediate dependency for the user.")
            if operational:
                school_admin_notice = sender_is_admin and group.get("group_type") == "school"
                action = "notify" if urgent or school_admin_notice else "digest"
                output_type = "urgent" if critical_operational else "event"
                return self._result(message, action, output_type, 0.84 if action == "notify" else 0.72, evidence_ids,
                                    "A group operational update is relevant to the user's near-term plans.")
            if category == "forward":
                return self._result(message, "mute", category, 0.78, evidence_ids,
                                    "This group message is routine social noise without a user-specific need.")
            if category == "greeting":
                return self._result(message, "digest", category, 0.68, evidence_ids,
                                    "This group greeting is harmless but does not require an interruption.")
            return self._result(message, "digest", category, 0.66, evidence_ids,
                                "This group message may be useful later but is not interrupt-worthy.")

        # Personal messages are valuable by default, with clear urgency promoted.
        if urgent:
            return self._result(message, "notify", "urgent", 0.83, evidence_ids,
                                "A personal sender indicates a time-sensitive request or deadline.")
        if category == "greeting":
            return self._result(message, "digest", "greeting", 0.68, evidence_ids,
                                "This personal greeting can be read without an immediate interruption.")
        if history_negative > history_positive:
            return self._result(message, "mute", category, 0.72, evidence_ids,
                                "The user has not found similar messages from this context useful.")
        return self._result(message, "digest", category, 0.69, evidence_ids,
                            "This personal message is safe and useful, but can wait for a digest.")

    @staticmethod
    def _result(message: dict[str, str], action: str, category: str, confidence: float,
                evidence_ids: str, reason: str) -> dict[str, str]:
        return {
            "message_id": message["message_id"], "action": action, "message_type": category,
            "reason": reason, "confidence": f"{confidence:.2f}", "evidence_message_ids": evidence_ids,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output.csv")
    args = parser.parse_args()
    router = Router(args.dataset)
    messages = read_csv(args.dataset / "messages.csv")
    predictions = [router.route(router.media.enrich(message)) for message in messages]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
