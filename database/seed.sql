-- ─────────────────────────────────────────────────────────────────────────────
-- database/seed.sql — Datos semilla iniciales
-- ─────────────────────────────────────────────────────────────────────────────
-- Se ejecuta automáticamente en el primer arranque del contenedor PostgreSQL
-- (via docker-entrypoint-initdb.d/02_seed.sql).
-- ─────────────────────────────────────────────────────────────────────────────

-- Verificar que el schema ya existe antes de insertar
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'taxa_class'
    ) THEN
        RAISE EXCEPTION 'Schema no inicializado. Ejecutar schema.sql primero.';
    END IF;
END $$;

-- ── Clases taxonómicas ────────────────────────────────────────────────────────
INSERT INTO taxa_class (name, common_name, acoustic_range_hz_low, acoustic_range_hz_high, notes)
VALUES
    ('Mammalia',  'Mamíferos',  20,      200000, 'Incluye murciélagos (ultrasónico) y mamíferos vocales'),
    ('Amphibia',  'Anfibios',   100,     8000,   'Principalmente anuros; cantos en rango audible'),
    ('Insecta',   'Insectos',   200,     100000, 'Ortópteros, cicadas; incluye componente ultrasónico'),
    ('Reptilia',  'Reptiles',   100,     5000,   'Crocodilianos y algunos geckos con vocalización'),
    ('Aves',      'Aves',       200,     12000,  'Canto de aves; módulo de expansión futura')
ON CONFLICT (name) DO NOTHING;

-- ── Dispositivos de grabación de referencia ───────────────────────────────────
INSERT INTO recording_device (
    model, manufacturer, max_sample_rate_hz,
    frequency_response_low_hz, frequency_response_high_hz,
    bit_depth, notes
)
VALUES
    ('AudioMoth v1.2',  'Open Acoustic Devices', 384000, 10,    384000, 16, 'Ultrasónico; ideal para murciélagos'),
    ('SM4BAT FS',       'Wildlife Acoustics',    384000, 10,    192000, 16, 'Estación pasiva para quirópteros'),
    ('SM4',             'Wildlife Acoustics',    48000,  20,    24000,  16, 'Monitoreo de aves y anfibios'),
    ('Zoom H5',         'Zoom',                  96000,  20,    40000,  24, 'Grabadora portátil de campo'),
    ('Rode NTG4+',      'Rode',                  48000,  75,    20000,  24, 'Micrófono shotgun para campo')
ON CONFLICT DO NOTHING;

-- ── Sitio de prueba por defecto (para desarrollo) ─────────────────────────────
INSERT INTO recording_site (
    name, country, state_province, locality,
    latitude, longitude, altitude_m, habitat_type, notes
)
VALUES
    (
        'Sitio de Prueba — Desarrollo',
        'México', 'Sin especificar', 'Dataset de desarrollo',
        19.432608, -99.133209, 2240,
        'urban',
        'Sitio ficticio para pruebas de desarrollo. Coordenadas: Ciudad de México.'
    )
ON CONFLICT DO NOTHING;

-- Confirmación
DO $$
DECLARE
    n_classes INT;
    n_devices INT;
BEGIN
    SELECT COUNT(*) INTO n_classes FROM taxa_class;
    SELECT COUNT(*) INTO n_devices FROM recording_device;
    RAISE NOTICE 'Seed completado: % clases taxonómicas, % dispositivos', n_classes, n_devices;
END $$;
