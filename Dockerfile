# Lightweight image for Render.com free tier (512 MB RAM).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

# Minimal TeX Live: no texlive-full, no docs, no recommends.
# cm-super-minimal provides Type1 Cyrillic (T2A) fonts, so pdflatex does not
# have to generate bitmap fonts at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-lang-cyrillic \
        texlive-science \
        cm-super-minimal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
              /usr/share/doc/* /usr/share/man/* /usr/share/info/* \
              /usr/share/texlive/texmf-dist/doc \
              /usr/share/texmf/doc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 bot
COPY --chown=bot:bot bot.py llm_service.py pdf_builder.py access.py ./
USER bot

# Fail the build early if the TeX installation cannot compile the template.
RUN python pdf_builder.py

EXPOSE 10000

CMD ["python", "bot.py"]
