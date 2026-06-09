-- ═══════════════════════════════════════════════════════════════════════════
-- SCHEMA: Sistema de Identificación de Fauna por Bioacústica + IA
-- DBMS  : PostgreSQL 15+
-- Autor : Ian
-- Versión: 1.0.0
-- ═══════════════════════════════════════════════════════════════════════════

-- Habilitar extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";    -- UUIDs
CREATE EXTENSION IF NOT EXISTS "postgis";      -- Coordenadas GPS
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Búsqueda por similitud de texto

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. TAXONOMÍA
-- ─────────────────────────────────────────────────────────────────────────────

-- Clases taxonómicas objetivo del sistema
CREATE TABLE taxa_class (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,   -- "Mammalia", "Amphibia", etc.
    common_name VARCHAR(100),
    description TEXT
);

-- Órdenes taxonómicos
CREATE TABLE taxa_order (
    id          SERIAL PRIMARY KEY,
    class_id    INT NOT NULL REFERENCES taxa_class(id) ON DELETE CASCADE,
    name        VARCHAR(100) NOT NULL,
    common_name VARCHAR(100),
    UNIQUE (class_id, name)
);

-- Familias taxonómicas
CREATE TABLE taxa_family (
    id          SERIAL PRIMARY KEY,
    order_id    INT NOT NULL REFERENCES taxa_order(id) ON DELETE CASCADE,
    name        VARCHAR(100) NOT NULL,
    common_name VARCHAR(100),
    UNIQUE (order_id, name)
);

-- Géneros
CREATE TABLE taxa_genus (
    id          SERIAL PRIMARY KEY,
    family_id   INT NOT NULL REFERENCES taxa_family(id) ON DELETE CASCADE,
    name        VARCHAR(100) NOT NULL,
    UNIQUE (family_id, name)
);

-- Especies — tabla central de referencia taxonómica
CREATE TABLE species (
    id              SERIAL PRIMARY KEY,
    genus_id        INT NOT NULL REFERENCES taxa_genus(id) ON DELETE RESTRICT,
    epithet         VARCHAR(100) NOT NULL,          -- epíteto específico
    authority       VARCHAR(200),                   -- "Linnaeus, 1758"
    common_name_es  VARCHAR(200),                   -- nombre común español
    common_name_en  VARCHAR(200),                   -- nombre común inglés
    iucn_status     VARCHAR(20)                     -- LC, NT, VU, EN, CR, EW, EX
        CHECK (iucn_status IN ('LC','NT','VU','EN','CR','EW','EX','DD','NE')),
    acoustic_group  VARCHAR(50) NOT NULL            -- mammal_bat, mammal_other,
        CHECK (acoustic_group IN (                  -- amphibian_anura, insect_orthoptera,
            'mammal_bat','mammal_other',            -- insect_cicada, reptile, bird, other
            'amphibian_anura','insect_orthoptera',
            'insect_cicada','reptile','bird','other'
        )),
    freq_min_hz     INT,       -- frecuencia mínima característica de vocalización
    freq_max_hz     INT,       -- frecuencia máxima
    freq_dom_hz     INT,       -- frecuencia dominante típica
    notes           TEXT,
    xeno_canto_id   VARCHAR(50),                    -- ID en Xeno-canto si aplica
    gbif_id         BIGINT,                         -- ID en GBIF
    inat_taxon_id   INT,                            -- ID en iNaturalist
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (genus_id, epithet)
);

-- Vista materializada: nombre científico completo
CREATE VIEW species_full AS
    SELECT
        s.id                   AS species_id,
        tg.name  || ' ' || s.epithet AS scientific_name,
        s.common_name_es,
        s.common_name_en,
        s.acoustic_group,
        s.iucn_status,
        tf.name                AS family,
        tord.name              AS "order",
        tc.name                AS class
    FROM species s
    JOIN taxa_genus  tg   ON tg.id  = s.genus_id
    JOIN taxa_family tf   ON tf.id  = tg.family_id
    JOIN taxa_order  tord ON tord.id = tf.order_id
    JOIN taxa_class  tc   ON tc.id  = tord.class_id;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. SITIOS Y EQUIPOS DE GRABACIÓN
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE recording_site (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(200) NOT NULL,
    country         VARCHAR(100),
    state_province  VARCHAR(100),
    locality        TEXT,
    latitude        DOUBLE PRECISION CHECK (latitude  BETWEEN -90  AND  90),
    longitude       DOUBLE PRECISION CHECK (longitude BETWEEN -180 AND 180),
    altitude_m      INT,
    habitat_type    VARCHAR(100),   -- "tropical rainforest", "wetland", etc.
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_site_coords ON recording_site (latitude, longitude);

CREATE TABLE recording_device (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,    -- "AudioMoth v1.2", "Zoom H5"
    model       VARCHAR(100),
    sample_rate INT,                      -- Hz máximo soportado
    bit_depth   SMALLINT,
    notes       TEXT
);


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. GRABACIONES ORIGINALES
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE recording (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id         UUID REFERENCES recording_site(id) ON DELETE SET NULL,
    device_id       INT  REFERENCES recording_device(id) ON DELETE SET NULL,
    filename        VARCHAR(500) NOT NULL,
    file_path       TEXT NOT NULL,              -- ruta relativa al data/ raíz
    format          VARCHAR(20) NOT NULL        -- "WAV", "MP3", "FLAC"
        CHECK (format IN ('WAV','MP3','FLAC','OGG','AIF')),
    sample_rate     INT NOT NULL,               -- Hz
    bit_depth       SMALLINT,
    channels        SMALLINT NOT NULL DEFAULT 1,
    duration_s      DOUBLE PRECISION NOT NULL,  -- segundos
    file_size_bytes BIGINT,
    recorded_at     TIMESTAMPTZ,                -- timestamp del inicio de la grabación
    temperature_c   DECIMAL(5,2),               -- temperatura ambiente
    humidity_pct    DECIMAL(5,2),               -- humedad relativa
    noise_level_db  DECIMAL(6,2),               -- nivel de ruido de fondo estimado
    checksum_sha256 CHAR(64),                   -- integridad del archivo
    source          VARCHAR(100)                -- "field", "xeno-canto", "inat", "macaulay"
        CHECK (source IN ('field','xeno-canto','inat','macaulay','gbif','frogid','batdb','other')),
    source_url      TEXT,
    license         VARCHAR(100),               -- "CC BY 4.0", "CC BY-SA 4.0", etc.
    quality_grade   CHAR(1)                     -- "A","B","C" (como Xeno-canto)
        CHECK (quality_grade IN ('A','B','C','D','E')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_rec_site     ON recording (site_id);
CREATE INDEX idx_rec_recorded ON recording (recorded_at);
CREATE INDEX idx_rec_source   ON recording (source);
CREATE INDEX idx_rec_quality  ON recording (quality_grade);


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. SEGMENTOS DE AUDIO PROCESADOS
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE audio_segment (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id    UUID NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    segment_index   INT  NOT NULL,
    t_start_s       DOUBLE PRECISION NOT NULL,  -- inicio en la grabación original
    t_end_s         DOUBLE PRECISION NOT NULL,  -- fin en la grabación original
    duration_s      DOUBLE PRECISION GENERATED ALWAYS AS (t_end_s - t_start_s) STORED,
    is_event        BOOLEAN NOT NULL DEFAULT FALSE,  -- True si es evento VAD detectado
    rms_energy      DECIMAL(10,6),                   -- energía RMS del segmento
    peak_amplitude  DECIMAL(10,6),                   -- amplitud pico
    snr_db          DECIMAL(8,3),                    -- relación señal-ruido estimada
    features_path   TEXT,                            -- ruta al .npy con features extraídas
    mel_path        TEXT,                            -- ruta al espectrograma Mel (.npy/.png)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (recording_id, segment_index)
);

CREATE INDEX idx_seg_recording ON audio_segment (recording_id);
CREATE INDEX idx_seg_event     ON audio_segment (is_event);


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. DATASET Y ETIQUETAS (GROUND TRUTH)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE dataset (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL UNIQUE,
    version     VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    description TEXT,
    split_type  VARCHAR(20) NOT NULL DEFAULT 'random'
        CHECK (split_type IN ('random','stratified','site-based','temporal')),
    train_pct   DECIMAL(5,2) DEFAULT 70.0,
    val_pct     DECIMAL(5,2) DEFAULT 15.0,
    test_pct    DECIMAL(5,2) DEFAULT 15.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Asignación de segmentos a splits del dataset
CREATE TABLE dataset_split (
    id          SERIAL PRIMARY KEY,
    dataset_id  INT  NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    segment_id  UUID NOT NULL REFERENCES audio_segment(id) ON DELETE CASCADE,
    split       VARCHAR(10) NOT NULL CHECK (split IN ('train','val','test')),
    UNIQUE (dataset_id, segment_id)
);

-- Etiquetas de especie sobre segmentos (ground truth)
CREATE TABLE segment_label (
    id              SERIAL PRIMARY KEY,
    segment_id      UUID NOT NULL REFERENCES audio_segment(id) ON DELETE CASCADE,
    species_id      INT  NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    confidence      DECIMAL(5,4) NOT NULL DEFAULT 1.0    -- confianza del anotador (0-1)
        CHECK (confidence BETWEEN 0 AND 1),
    label_type      VARCHAR(20) NOT NULL DEFAULT 'expert'
        CHECK (label_type IN ('expert','citizen','model','weak')),
    annotator       VARCHAR(200),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (segment_id, species_id)
);

CREATE INDEX idx_label_segment ON segment_label (segment_id);
CREATE INDEX idx_label_species ON segment_label (species_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. MODELOS DE CLASIFICACIÓN
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE ml_model (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    version         VARCHAR(50)  NOT NULL,
    architecture    VARCHAR(100) NOT NULL,   -- "EfficientNet-B0", "ResNet-50", "CNN14-PANNs"
    input_type      VARCHAR(50)  NOT NULL    -- "mel_spectrogram", "mfcc", "raw_audio"
        CHECK (input_type IN ('mel_spectrogram','mfcc','combined','raw_audio','embedding')),
    n_classes       INT NOT NULL,
    acoustic_groups TEXT[],                 -- grupos taxonómicos cubiertos
    dataset_id      INT REFERENCES dataset(id) ON DELETE SET NULL,
    weights_path    TEXT,                   -- ruta al archivo .pt / .h5
    config_path     TEXT,                   -- ruta al YAML de configuración
    -- Métricas de evaluación (test set)
    accuracy        DECIMAL(7,4),
    f1_macro        DECIMAL(7,4),
    precision_macro DECIMAL(7,4),
    recall_macro    DECIMAL(7,4),
    auc_roc_macro   DECIMAL(7,4),
    -- Metadatos de entrenamiento
    framework       VARCHAR(50)             -- "pytorch", "tensorflow", "sklearn"
        CHECK (framework IN ('pytorch','tensorflow','sklearn','onnx','other')),
    epochs_trained  INT,
    batch_size      INT,
    learning_rate   DECIMAL(10,8),
    notes           TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,  -- modelo en producción
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. DETECCIONES (INFERENCIA)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE detection (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    segment_id      UUID NOT NULL REFERENCES audio_segment(id) ON DELETE CASCADE,
    model_id        INT  NOT NULL REFERENCES ml_model(id) ON DELETE RESTRICT,
    species_id      INT  REFERENCES species(id) ON DELETE SET NULL,
    probability     DECIMAL(7,6) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    rank            SMALLINT NOT NULL DEFAULT 1,    -- posición en top-K predicciones
    is_correct      BOOLEAN,                        -- NULL = sin verificación
    verified_by     VARCHAR(200),
    verified_at     TIMESTAMPTZ,
    inference_ms    INT,                            -- tiempo de inferencia en ms
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_det_segment   ON detection (segment_id);
CREATE INDEX idx_det_model     ON detection (model_id);
CREATE INDEX idx_det_species   ON detection (species_id);
CREATE INDEX idx_det_prob      ON detection (probability DESC);
CREATE INDEX idx_det_created   ON detection (created_at DESC);

-- Vista de detecciones con contexto completo
CREATE VIEW detection_summary AS
    SELECT
        d.id                    AS detection_id,
        r.filename              AS recording,
        rs.name                 AS site,
        rs.latitude, rs.longitude,
        aseg.t_start_s, aseg.t_end_s,
        sf.scientific_name,
        sf.common_name_es,
        sf.acoustic_group,
        d.probability,
        d.rank,
        d.is_correct,
        mm.name  || ' v' || mm.version AS model,
        d.created_at
    FROM detection d
    JOIN audio_segment aseg ON aseg.id       = d.segment_id
    JOIN recording      r   ON r.id          = aseg.recording_id
    LEFT JOIN recording_site rs ON rs.id     = r.site_id
    JOIN ml_model       mm  ON mm.id         = d.model_id
    LEFT JOIN species_full sf ON sf.species_id = d.species_id;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. REPORTES DE BIODIVERSIDAD
-- ─────────────────────────────────────────────────────────────────────────────

-- Resumen de riqueza acústica por sitio y período
CREATE VIEW site_biodiversity_report AS
    SELECT
        rs.id                           AS site_id,
        rs.name                         AS site_name,
        rs.country,
        DATE_TRUNC('month', r.recorded_at) AS month,
        COUNT(DISTINCT d.species_id)    AS species_richness,
        COUNT(DISTINCT sf.acoustic_group) AS acoustic_groups,
        COUNT(d.id)                     AS total_detections,
        AVG(d.probability)              AS avg_confidence,
        ARRAY_AGG(DISTINCT sf.scientific_name ORDER BY sf.scientific_name)
                                        AS species_list
    FROM detection d
    JOIN audio_segment aseg ON aseg.id = d.segment_id
    JOIN recording      r   ON r.id    = aseg.recording_id
    JOIN recording_site rs  ON rs.id   = r.site_id
    LEFT JOIN species_full sf ON sf.species_id = d.species_id
    WHERE d.probability >= 0.80
      AND d.rank = 1
    GROUP BY rs.id, rs.name, rs.country, DATE_TRUNC('month', r.recorded_at);


-- ─────────────────────────────────────────────────────────────────────────────
-- 9. DATOS INICIALES (SEED)
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO taxa_class (name, common_name) VALUES
    ('Mammalia',  'Mamíferos'),
    ('Amphibia',  'Anfibios'),
    ('Reptilia',  'Reptiles'),
    ('Insecta',   'Insectos'),
    ('Aves',      'Aves');

INSERT INTO recording_device (name, model, sample_rate, bit_depth) VALUES
    ('AudioMoth',       'Open Acoustic Devices AudioMoth v1.2', 384000, 16),
    ('Zoom H5',         'Zoom H5 Handy Recorder',               96000,  24),
    ('Sony PCM-A10',    'Sony PCM-A10',                         192000, 24),
    ('Raspberry Pi HAT','Pi NoIR + MEMS INMP441',               48000,  32),
    ('Pettersson D500x','Pettersson D500x',                      500000, 16);

INSERT INTO dataset (name, version, description, split_type) VALUES
    ('BioAcoustics-MultiTaxa-v1', '1.0.0',
     'Dataset multitaxonómico inicial: mamíferos (murciélagos + otros), anfibios anuros, insectos acústicos, reptiles. Fuentes: Xeno-canto, iNaturalist, FrogID, campo propio.',
     'stratified');
