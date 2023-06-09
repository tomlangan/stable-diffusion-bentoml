#!/bin/bash
BENTOML_CONFIG="configuration.yaml"
bentoml serve service:svc --production --port 7070
