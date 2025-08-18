import re

def extract_sql_only(text: str) -> str:
    """
    LLM bazen açıklama/markdown döndürüyor. Bu fonksiyon sadece çalıştırılabilir
    SQL'i bırakır. ```sql ... ``` varsa içini, yoksa tüm metni strip eder.
    """
    if not text:
        return ""
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(";")
    # Kod bloğu yoksa yine de sadeleştir
    # Bazı modeller başa/sona metin ekliyor; ilk SELECT/WITH'ten kırp
    m2 = re.search(r"\b(WITH|SELECT)\b(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    return (m2.group(0) if m2 else text).strip().rstrip(";")
