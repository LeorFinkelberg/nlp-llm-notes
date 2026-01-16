RAG (retrieval augmented generation) -- генерация дополненная поиском. RAG-системы обычно имеют 3 основных компонента: ретривер (поисковый модуль), аугментатор (обогащает промпт) и генератор (генерация ответа).

Типы архитектур RAG:
- Простой RAG (Naive RAG)
```python
# Базовая версия
def naive_rag(query):
    docs = retrieve(query)      # Поиск
    answer = generate(docs)     # Генерация
    return answer
```
- Продвинутый RAG (Advanced RAG)
```python
def advanced_rag(query):
    # Переформулировка запроса
    rewritten_query = query_rewriter(query)
    
    # Поиск в нескольких проходах
    docs = []
    for subquery in split_query(rewritten_query):
        docs.extend(hybrid_search(subquery))  # Гибридный поиск
        
    # Переранжирование результатов
    docs = rerank(docs, query)
    
    # Извлечение наиболее релевантных частей
    context = extract_relevant_chunks(docs, query)
    
    # Генерация с проверкой фактов
    answer = generate_with_verification(context, query)
    
    return answer
```
- Модульный RAG (Modular RAG)
```python
class ModularRAG:
    def __init__(self):
        self.modules = {
            "query_understanding": QueryUnderstandingModule(),
            "search": MultiSourceSearchModule(),
            "reranking": RerankingModule(),
            "fusion": InformationFusionModule(),
            "generation": FactCheckingGenerationModule(),
        }
    
    def answer(self, query):
        results = {}
        for name, module in self.modules.items():
            results[name] = module.process(query, results)
        return results["generation"]
```
- Саморефлексивый RAG (Self-RAG): модель сама решает нужен ли поиск, как запрос использовать для поиска, достаточно ли найденной информации и как сгенерировать ответ.
- Мульти-запросный поиск (RAG-Fusion).

Ключевая ценность RAG заключается в том, что она позволяет LLM "знать то, чего она не знала" БЕЗ ПЕРЕОБУЧЕНИЯ, просто предоставляя актуальные документы в контексте запроса.

