ARG WILDCLAW_BASE=ghcr.io/valen-team/roller-wildclaw-base:1.3-clean.1
FROM ghcr.io/valen-team/roller-codex:0.149.0 AS codex
FROM ${WILDCLAW_BASE}
COPY --from=codex /usr/local/lib/node_modules/@openai/codex /opt/roller-codex
COPY --from=codex /usr/local/bin/node /usr/local/bin/node
COPY --from=codex /usr/bin/rg /usr/local/bin/rg
RUN ln -sf /opt/roller-codex/bin/codex.js /usr/local/bin/codex \
 && bin="$(find /opt/roller-codex -type f -path '*/bin/codex' -print -quit)" \
 && test -x "$bin" && ln -sf "$bin" /usr/local/bin/apply_patch \
 && codex --version && python3 --version
COPY --from=codex /usr/local/LICENSE /usr/share/doc/roller-wildclaw/LICENSE-Node
COPY --from=codex /usr/share/doc/ripgrep/copyright /usr/share/doc/roller-wildclaw/LICENSE-ripgrep
COPY LICENSE-Codex /usr/share/doc/roller-wildclaw/LICENSE-Codex
COPY THIRD_PARTY_NOTICES /usr/share/doc/roller-wildclaw/THIRD_PARTY_NOTICES
LABEL org.opencontainers.image.source="https://github.com/Valen-Team/Roller" \
      org.opencontainers.image.version="1.3-codex0.149.0" \
      org.opencontainers.image.description="WildClawBench task environment with Codex CLI 0.149.0"
ENV CODEX_CI=1 DISABLE_AUTOUPDATER=1
WORKDIR /tmp_workspace
CMD ["bash"]
