# The environment VIALS lists for its tool-assisted agent: Python 3.13, the
# eleven Python packages named in its prompt, and the Tesseract OCR engine.
#   docker build -t roller-vials docker/vials
FROM python:3.13-slim

ARG DEBIAN_MIRROR
ARG DEBIAN_SECURITY_MIRROR

# Match complete repository URLs so the main mirror cannot rewrite security URLs.
# Keep this block in each build context; the contexts are built independently.
RUN set -eu; \
    for source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "$source" ] || continue; \
        if [ -n "${DEBIAN_MIRROR:-}" ]; then \
            mirror=$(printf '%s' "${DEBIAN_MIRROR%/}" | sed 's/[\\&@]/\\&/g'); \
            sed -i -E "s@https?://deb[.]debian[.]org/debian/?([[:space:]]|$)@$mirror\\1@g" "$source"; \
        fi; \
        if [ -n "${DEBIAN_SECURITY_MIRROR:-}" ]; then \
            mirror=$(printf '%s' "${DEBIAN_SECURITY_MIRROR%/}" | sed 's/[\\&@]/\\&/g'); \
            sed -i -E "s@https?://(deb|security)[.]debian[.]org/debian-security/?([[:space:]]|$)@$mirror\\2@g" "$source"; \
        fi; \
    done

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action ignore -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1
