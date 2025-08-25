# Customer Persona Pipeline 🧑‍🤝‍🧑

This repo extracts YouTube comments → embeds them → clusters them → generates **customer personas**.

## Workflow
1. `questions.py` → initial business questions
2. `findcomp.py` → competitor list
3. `followup.py` → follow-up questions per competitor
4. `youtube.py` → fetch top videos & comments
5. `pipeline.py` → run:
   - `embeddings.py` (vectorize comments)
   - `clustering.py` (group into themes)
   - `persona_gen.py` (LLM or heuristic → structured personas)

## Run

```bash
# activate venv
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# step 1–4: produce youtube_analysis.json
python questions.py
python findcomp.py
python followup.py
python youtube.py

# step 5: full pipeline
python pipeline.py --youtube youtube_analysis.json
