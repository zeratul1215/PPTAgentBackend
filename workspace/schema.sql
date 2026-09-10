-- PPTAgent application schema (metadata + session state).
--
-- This holds the *structured* state that must survive process restarts:
--   * users            — ownership root.
--   * decks            — per-deck metadata (replaces scanning result/*/project.json).
--   * sessions         — chat threads + which deck each is currently editing.
--   * session_decks    — per-session recently-used decks, ordered.
--   * turns            — single-page edit history (metadata only).
--
-- Large artifacts (PNGs, HTML bundle, source PDF/PPTX, gpt-image-2 refs, page
-- state JSON) stay on disk / object storage; only references live here.
--
-- LangGraph's PostgresSaver owns the checkpoint tables (checkpoints,
-- checkpoint_writes, ...) and creates them via its own .setup(); we do NOT
-- define them here.

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,          -- opaque uuid
    username     TEXT NOT NULL,             -- login handle (what the user types)
    display_name TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Passwordless login: the same username always resolves to the same account.
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(lower(username));

CREATE TABLE IF NOT EXISTS decks (
    project_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    page_count    INT  NOT NULL DEFAULT 0,
    page_size_pt  JSONB,
    source_kind   TEXT,               -- 'pdf' | 'pptx' | 'ppt' | ...
    source_path   TEXT,               -- original upload location (local for now)
    workspace_root TEXT,              -- result/<project_id>/ on disk
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_decks_user ON decks(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    active_project_id TEXT REFERENCES decks(project_id) ON DELETE SET NULL,
    title             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, last_active_at DESC);

-- Recently-used decks per session (replaces the in-memory recent_project_ids
-- list). Ordered by last_used_at for "the previous deck" references.
CREATE TABLE IF NOT EXISTS session_decks (
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    project_id   TEXT NOT NULL REFERENCES decks(project_id) ON DELETE CASCADE,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, project_id)
);
CREATE INDEX IF NOT EXISTS idx_session_decks_recent
    ON session_decks(session_id, last_used_at DESC);

-- Single-page edit history. One edit_pages call for one page = one turn.
-- step artifacts (step2_output.json / html) stay on disk under turns/turn_xxxx/;
-- here we index the metadata for "how many times / rollback / audit".
CREATE TABLE IF NOT EXISTS turns (
    turn_id          BIGSERIAL PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES decks(project_id) ON DELETE CASCADE,
    session_id       TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    page_num         INT  NOT NULL,          -- 1-based
    demand           TEXT NOT NULL,          -- the standalone instruction for this page
    ok               BOOLEAN NOT NULL DEFAULT TRUE,
    error            TEXT,
    turn_dir         TEXT,                   -- turns/turn_xxxx/ on disk
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_turns_page ON turns(project_id, page_num, created_at DESC);

CREATE TABLE IF NOT EXISTS deck_styles (
    project_id       TEXT PRIMARY KEY REFERENCES decks(project_id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'failed',
    style_json       JSONB,
    source           TEXT,
    revision         INT NOT NULL DEFAULT 1,
    analysis_run_id  TEXT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deck_styles_status ON deck_styles(status);

CREATE TABLE IF NOT EXISTS deck_style_presets (
    preset_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    style_json  JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deck_style_presets_user
    ON deck_style_presets(user_id, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_deck_style_presets_user_name_unique
    ON deck_style_presets(user_id, lower(btrim(name)));

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id        TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id           TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system_event')),
    content           TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'complete',
    client_message_id TEXT,
    active_project_id TEXT REFERENCES decks(project_id) ON DELETE SET NULL,
    selected_slot     INT,
    page_order_revision INT,
    attachments       JSONB NOT NULL DEFAULT '[]'::jsonb,
    seq               BIGSERIAL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_seq
    ON chat_messages(session_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_idempotency
    ON chat_messages(session_id, client_message_id)
    WHERE client_message_id IS NOT NULL AND role = 'user';

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id       TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id            TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    user_message_id    TEXT NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    assistant_message_id TEXT NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    active_project_id  TEXT REFERENCES decks(project_id) ON DELETE SET NULL,
    selected_slot      INT,
    page_order_revision INT,
    status             TEXT NOT NULL CHECK (status IN ('queued', 'running', 'waiting', 'complete', 'failed', 'cancelled')),
    todos              JSONB NOT NULL DEFAULT '[]'::jsonb,
    gate               JSONB,
    error              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at       TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_one_active_per_session
    ON agent_runs(session_id)
    WHERE status IN ('queued', 'running', 'waiting');
CREATE INDEX IF NOT EXISTS idx_agent_runs_session_created
    ON agent_runs(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_run_events (
    event_id     BIGSERIAL PRIMARY KEY,
    agent_run_id TEXT NOT NULL REFERENCES agent_runs(agent_run_id) ON DELETE CASCADE,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_run_events_run_cursor
    ON agent_run_events(agent_run_id, event_id);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id      TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    covered_seq     BIGINT NOT NULL DEFAULT 0,
    version         INT NOT NULL DEFAULT 1,
    summary         TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS session_states (
    session_id          TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id             TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    revision            INT NOT NULL DEFAULT 1,
    updated_through_seq BIGINT NOT NULL DEFAULT 0,
    entries             JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_ref     TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id       TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id       TEXT REFERENCES chat_messages(message_id) ON DELETE SET NULL,
    filename         TEXT NOT NULL,
    mime             TEXT NOT NULL,
    size_bytes       BIGINT NOT NULL DEFAULT 0,
    width            INT,
    height           INT,
    sha256           TEXT NOT NULL,
    storage_path     TEXT NOT NULL,
    thumbnail_path   TEXT,
    status           TEXT NOT NULL CHECK (status IN ('draft', 'attached', 'deleted')) DEFAULT 'draft',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_session_created
    ON artifacts(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_draft_cleanup
    ON artifacts(status, created_at);
