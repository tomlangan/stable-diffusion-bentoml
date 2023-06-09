$env:BENTOML_CONFIG = "configuration.yaml"
conda activate stablediff
bentoml serve service:svc --production --port 7070

