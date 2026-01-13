# Engifi O*NET Role Mapper

Goal: map a company's job title + description to O*NET-SOC occupation code(s), with confidence scores.

## Quickstart
1) Create venv and install deps:
   - pip install -r requirements.txt -r requirements-dev.txt
   - pip install -e .

2) Configure:
   - copy .env.example to .env
   - set ONET_API_KEY=...

3) Run:
   - engifi-onet info
   - engifi-onet ping

## Attribution
O*NET Database content is available under CC BY 4.0 and requires attribution in downstream usage.
