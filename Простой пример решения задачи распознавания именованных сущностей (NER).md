Рассмотрим решение простой _задачи распознавания именованных сущностей_ (Named Entity Recognition, NER). На странице платформы HaggingFace находим модель BERT-BASE-NER https://huggingface.co/dslim/bert-base-NER. BERT-BASE-NER -- это настроенная BERT-модель, готовая для решения NER-задач. Она была обучена распознавать 4 типа сущностей: локации (LOC), организации (ORG), персоны (PER) и прочее (MISC).

На этой же странице можно найти инструкции по работе с моделью. Предварительно требуется установить библиотеки PyTorch (детали по установке под различные ОС можно найти [здесь](https://pytorch.org/get-started/locally/)) и Transformers
```bash
# Установка Torch на Linux с поддержкой только CPU
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install transformers
```
После установки библиотек, приступить к решению задачи можно так
```python
from enum import Enum, auto
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class _UpperCaseAttrs(Enum):
    @staticmethod
    def _genereate_next_value_(name, start, count, last_values):
      return name.upper()

class EntityType(_UpperCaseAttr):
    O = auto()
    MISC = auto()
    PER = auto()
    ORG = auto()
    LOC = auto()

def get_words_by_entity_type(
	entities: t.List[t.Dict],
	entity_type: EntityType,
) -> t.List[str]:
    entity_type: str = entity_type.value
    return [
        block["word"] for block in entities
        if entity_type in block["entity"]
    ]

MODEL_NAME = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

message = (
	"My name is Leor and I live in Russia. "
	"And my girlfriend Julia live in Brazil"
)
get_words_by_entity_type(
	nlp(message),
	entity_type=EntityType.LOC
)  # ['Russia', 'Brazil']

message = "I like Python language! And a little more Java language"
get_words_by_entity_type(
	nlp(message),
	entity_type=EntityType.MISC
)  # ['Python', 'Java']
```