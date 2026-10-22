from app.services.zep_tools import ZepToolsService
from app.utils.locale import set_locale


def test_sub_queries_follow_the_report_language():
    prompts = []

    class Llm:
        def chat_json(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return {"sub_queries": ["q"]}

    service = object.__new__(ZepToolsService)
    service._llm_client = Llm()
    set_locale("en")
    try:
        service._generate_sub_queries("query", "requirement")
    finally:
        set_locale("zh")

    # Chinese sub-queries pulled English report sections into Chinese.
    assert "Please respond in English." in prompts[0]
