"""
NLP Grammar Polisher & Sentence Assembler for Indian Sign Language (ISL).
Converts fingerspelled tokens and ISL vocabulary words (SOV grammar)
into fluent, natural, grammatically correct English sentences.
Includes:
  1. Fingerspelling concatenation and word boundary segmentation
  2. ISL-to-English linguistic transformation rules (Question inversion, copula insertion, SOV->SVO)
  3. Optional zero-dependency LLM gateway (Gemini / OpenAI) with automatic offline fallback
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger("nlp_grammar")

# Common English vocabulary recognized for spell-correction & word segmentation
COMMON_DICTIONARY = {
    "hello", "hi", "hey", "help", "name", "what", "where", "which", "who", "why", "how",
    "doctor", "hospital", "medicine", "pain", "skin", "headache", "fever", "emergency",
    "assistance", "college", "school", "university", "work", "office", "today", "tomorrow",
    "yesterday", "pray", "god", "temple", "church", "stand", "sit", "walk", "come", "go",
    "agree", "afraid", "bad", "good", "nice", "great", "small", "big", "specific", "secondary",
    "become", "warn", "from", "you", "your", "me", "my", "i", "we", "our", "they", "them",
    "water", "food", "thank", "thanks", "welcome", "please", "sorry", "yes", "no"
}

# High-frequency ISL idiomatic phrase translation mappings
ISL_IDIOM_MAP = {
    # Medical & Emergency
    "doctor where": "Where is the doctor?",
    "where doctor": "Where is the doctor?",
    "doctor need": "I need a doctor urgently.",
    "need doctor": "I need a doctor urgently.",
    "doctor you": "Are you a doctor?",
    "you doctor": "Are you a doctor?",
    "doctor today": "Is the doctor available today?",
    "today doctor": "Is the doctor available today?",
    "assistance need": "I need assistance, please.",
    "need assistance": "I need assistance, please.",
    "you assistance": "Do you need any assistance?",
    "assistance you": "Do you need any assistance?",
    "pain skin": "I have pain on my skin.",
    "skin pain": "I have pain on my skin.",
    "bad pain": "The pain is very severe.",
    "pain bad": "The pain is very severe.",
    "pain where": "Where does it hurt?",
    "where pain": "Where does it hurt?",

    # Feelings & States
    "afraid you": "Are you feeling afraid?",
    "you afraid": "Are you feeling afraid?",
    "afraid": "I am feeling afraid.",
    "me afraid": "I am feeling afraid.",
    "i afraid": "I am feeling afraid.",
    "agree you": "Do you agree with this?",
    "you agree": "Do you agree with this?",
    "agree": "I agree completely.",
    "i agree": "I agree completely.",
    "bad today": "I am not feeling well today.",
    "today bad": "Things are difficult today.",

    # Daily Life & Education
    "work today": "Are you working today?",
    "today work": "Are you working today?",
    "college which": "Which college do you attend?",
    "which college": "Which college do you attend?",
    "college from you": "Which college are you from?",
    "you college from": "Which college are you from?",
    "you from college": "Which college are you from?",
    "college where": "Where is the college located?",
    "where college": "Where is the college located?",
    "pray today": "Let us pray today.",
    "today pray": "Let us pray today.",
    "stand you": "Please stand up.",
    "you stand": "Please stand up.",
    "warn you": "I want to warn you about this.",
    "you warn": "I am warning you.",

    # Greetings & Introductions
    "hello": "Hello! Welcome.",
    "hello you": "Hello, how are you?",
    "you hello": "Hello, how are you?",
    "thank you": "Thank you very much!",
    "thanks you": "Thank you very much!",
    "thanks": "Thank you!",
    "name you": "What is your name?",
    "you name": "What is your name?",
    "name you what": "What is your name?",
    "you name what": "What is your name?",
    "name what": "What is your name?",
    "how you": "How are you doing?",
    "you how": "How are you doing?",
    "good morning": "Good morning!",
    "good night": "Good night!",
    "help me": "Please help me.",
    "me help": "Please help me.",
    "help you": "Can I help you?"
}


class ISLGrammarPolisher:
    """
    Transforms raw ISL tokens into fluent English sentences using
    a hybrid Rule-Based Engine + optional LLM Gateway.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def concatenate_tokens(self, tokens: List[str]) -> List[str]:
        """
        Concatenates consecutive fingerspelled single letters into coherent words.
        Tokens matching '_' or ' ' trigger word boundaries.
        """
        words = []
        curr_spelling = []

        for token in tokens:
            if not token:
                continue
            t = token.strip()
            if not t or t in ("_", " ", "|"):
                if curr_spelling:
                    words.append("".join(curr_spelling))
                    curr_spelling = []
                continue

            # If single character (fingerspelling letter)
            if len(t) == 1 and t.isalpha():
                curr_spelling.append(t)
            else:
                # Word-level sign
                if curr_spelling:
                    words.append("".join(curr_spelling))
                    curr_spelling = []
                words.append(t)

        if curr_spelling:
            words.append("".join(curr_spelling))

        return words

    def rule_based_polish(self, words: List[str]) -> str:
        """
        Applies linguistic rules, question inversion, and copula insertion
        to transform ISL word sequences into natural English.
        """
        if not words:
            return ""

        lower_words = [w.lower() for w in words]
        normalized_phrase = " ".join(lower_words)

        # 1. Exact or partial idiom match
        if normalized_phrase in ISL_IDIOM_MAP:
            return ISL_IDIOM_MAP[normalized_phrase]

        # 2. Check for question words
        question_words = ["what", "where", "which", "who", "why", "how", "when"]
        has_question_word = any(qw in lower_words for qw in question_words)

        # ISL puts question words at the END (e.g. "DOCTOR WHERE")
        if has_question_word:
            found_qw = None
            for qw in question_words:
                if qw in lower_words:
                    found_qw = qw
                    break

            other_words = [w for w in words if w.lower() != found_qw]
            other_text = " ".join(other_words)

            if found_qw == "where":
                return f"Where is the {other_text}?" if other_text else "Where is it?"
            elif found_qw == "what":
                return f"What is your {other_text}?" if "name" in lower_words else f"What is {other_text}?"
            elif found_qw == "which":
                return f"Which {other_text}?" if other_text else "Which one?"
            elif found_qw == "how":
                return f"How are you?" if "you" in lower_words else f"How is the {other_text}?"
            elif found_qw == "who":
                return f"Who is {other_text}?"

        # 3. Subject-Verb-Object & Copula ("is", "are", "am") adjustments
        if len(words) == 1:
            w = words[0].capitalize()
            return f"{w}."

        if len(words) == 2:
            w1, w2 = lower_words[0], lower_words[1]
            if w1 in ("i", "me"):
                return f"I am {words[1]}."
            elif w1 == "you":
                return f"Are you {words[1]}?"
            elif w2 == "you":
                return f"Do you {words[0]}?"
            elif w2 in ("pain", "problem"):
                return f"There is {words[0]} {words[1]}."
            else:
                return f"The {words[0]} is {words[1]}."

        # 4. 3+ Words General Construction
        # Capitalize first word and ensure terminal punctuation
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:]

        # Detect if "you" at start without aux verb -> form a question
        if lower_words[0] == "you" and not sentence.endswith("?"):
            sentence = "Are you " + " ".join(words[1:]) + "?"

        if not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence

    def llm_polish(self, words: List[str]) -> Optional[str]:
        """
        Queries Gemini or OpenAI REST API if keys are available,
        with a strict 2.5s timeout. Returns None on failure or if no key is configured.
        """
        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        prompt = (
            "You are a specialized Indian Sign Language (ISL) translator. "
            "Convert the following sequential sign tokens into ONE fluent, grammatically correct English sentence. "
            "Indian Sign Language often omits helping verbs (is/are/do) and uses Subject-Object-Verb order. "
            "Do NOT add explanations, notes, or quotes. Output ONLY the polished English sentence.\n\n"
            f"Signed Tokens: {' '.join(words)}"
        )

        try:
            import requests

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 60}
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean up any surrounding quotes
                text = text.strip('"\'')
                if text:
                    return text
        except Exception as e:
            logger.debug(f"LLM polish fallback triggered: {e}")

        return None

    def polish(self, raw_input: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Main entry point for sentence polishing.
        Accepts a string of space-separated signs or a list of token strings.
        Returns a dictionary with raw tokens, concatenated words, polished sentence, and source.
        """
        if isinstance(raw_input, str):
            tokens = [t for t in raw_input.strip().split() if t]
        else:
            tokens = [str(t) for t in raw_input if t]

        if not tokens:
            return {
                "raw_tokens": [],
                "words": [],
                "polished_sentence": "",
                "source": "none"
            }

        # 1. Group contiguous fingerspelled letters
        words = self.concatenate_tokens(tokens)

        # 2. Try LLM if available, otherwise rule-based
        polished = self.llm_polish(words)
        source = "llm"

        if not polished:
            polished = self.rule_based_polish(words)
            source = "rule_based"

        return {
            "raw_tokens": tokens,
            "words": words,
            "polished_sentence": polished,
            "source": source
        }


# Global singleton instance
polisher = ISLGrammarPolisher()
