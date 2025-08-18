# tools/visualize_sql_langraph.py
import os
import sys

# Proje kökünden çalıştırdığından emin ol (gerekirse sys.path ekleyebilirsin)
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.agents.sql.sql_langraph import graph_main

OUT = os.getenv("GRAPH_OUT", "sql_langraph.png")

def main():
    g = graph_main.get_graph()
    # 1) PNG dene
    try:
        png_bytes = g.draw_png()  # graphviz + pydot gerektirir
        with open(OUT, "wb") as f:
            f.write(png_bytes)
        print(f"[ok] Graph PNG kaydedildi: {OUT}")
        return
    except Exception as e:
        print(f"[warn] PNG render başarısız: {e}")

    # 2) Mermaid'e düş
    try:
        mermaid = g.draw_mermaid()
        out_mmd = os.path.splitext(OUT)[0] + ".mmd"
        with open(out_mmd, "w", encoding="utf-8") as f:
            f.write(mermaid)
        print(f"[ok] Mermaid dosyası kaydedildi: {out_mmd}")
        print("     Bunu https://mermaid.live ile açıp görselleştirebilirsin.")
        return
    except Exception as e2:
        print(f"[warn] Mermaid render başarısız: {e2}")

    # 3) Son çare: ASCII
    try:
        print("[info] ASCII graf:")
        print(g.draw_ascii())
    except Exception as e3:
        print(f"[err] ASCII render da başarısız: {e3}")

if __name__ == "__main__":
    main()
