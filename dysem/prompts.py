"""PromptEOL prompt templates used by DyDim."""

from .languages import DEFAULT_LANGUAGE


PROMPTEOL_ENGLISH_TEMPLATE = 'This sentence : "*sent 0*" means in one word:"'

LANGUAGE_SPECIFIC_PROMPTEOL_TEMPLATES = {
    "eng_Latn": PROMPTEOL_ENGLISH_TEMPLATE,
    "zho_Hans": '这句话：“*sent 0*”用一个词来表达就是：“',
    "fra_Latn": 'Cette phrase : "*sent 0*" signifie en un mot :"',
    "deu_Latn": 'Dieser Satz : "*sent 0*" bedeutet in einem Wort :"',
    "spa_Latn": 'Esta oración : "*sent 0*" significa en una palabra :"',
    "rus_Cyrl": 'Это предложение : "*sent 0*" означает одним словом :"',
    "arb_Arab": 'هذه الجملة : "*sent 0*" تعني بكلمة واحدة :"',
    "jpn_Jpan": 'この文 : 「*sent 0*」を一言で表すと :「',
    "kor_Hang": '이 문장 : "*sent 0*"을 한 단어로 표현하면 :"',
    "ita_Latn": 'Questa frase : "*sent 0*" significa in una parola :"',
    "por_Latn": 'Esta frase : "*sent 0*" significa em uma palavra :"',
    "hin_Deva": 'यह वाक्य : "*sent 0*" का एक शब्द में अर्थ है :"',
}


def normalize_sentence(sentence: str) -> str:
    """Normalize a sentence before it is inserted into PromptEOL."""
    normalized = str(sentence).strip()
    if normalized and normalized[-1] not in '.?"\'':
        normalized += "."
    normalized = normalized.replace('"', "'")
    if normalized.endswith("?"):
        normalized = normalized[:-1] + "."
    return normalized


def apply_prompteol_prompt(sentence: str, prompt_setting: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Apply either the English or language-specific PromptEOL template."""
    prompt_setting = str(prompt_setting)
    if prompt_setting == "english":
        template = PROMPTEOL_ENGLISH_TEMPLATE
    elif prompt_setting == "language-specific":
        try:
            template = LANGUAGE_SPECIFIC_PROMPTEOL_TEMPLATES[language]
        except KeyError as exc:
            raise ValueError(f"No language-specific prompteol template for {language!r}.") from exc
    else:
        raise ValueError(f"Unsupported prompt setting: {prompt_setting!r}.")

    return template.replace("*sent 0*", normalize_sentence(sentence)).strip()


def apply_prompts(texts: list[str], prompt_setting: str, languages: list[str]) -> list[str]:
    """Vectorized prompt application with one language code per text."""
    if len(texts) != len(languages):
        raise ValueError("texts and languages must have the same length.")
    return [
        apply_prompteol_prompt(text, prompt_setting=prompt_setting, language=language)
        for text, language in zip(texts, languages)
    ]
