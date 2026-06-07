-- ─────────────────────────────────────────────────────────────────────────────
-- database/seed.sql -- Datos semilla compatibles con database/schema.sql
-- Perfil inicial: aves de Mexico para identificacion bioacustica.
-- ─────────────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'taxa_class'
    ) THEN
        RAISE EXCEPTION 'Schema no inicializado. Ejecutar schema.sql primero.';
    END IF;
END $$;

-- ── Clases taxonomicas base ─────────────────────────────────────────────────
INSERT INTO taxa_class (name, common_name, description)
VALUES
    ('Mammalia', 'Mamiferos', 'Mamiferos vocales y quiropteros para expansion futura.'),
    ('Amphibia', 'Anfibios', 'Principalmente anuros con cantos en rango audible.'),
    ('Reptilia', 'Reptiles', 'Reptiles vocales como geckos y crocodilianos.'),
    ('Insecta', 'Insectos', 'Insectos acusticos como ortopteros y cicadas.'),
    ('Aves', 'Aves', 'Perfil inicial del sistema: aves presentes en Mexico.')
ON CONFLICT (name) DO UPDATE
SET common_name = EXCLUDED.common_name,
    description = EXCLUDED.description;

-- ── Dispositivos de referencia ──────────────────────────────────────────────
WITH device_data(name, model, sample_rate, bit_depth, notes) AS (
    VALUES
        ('AudioMoth', 'Open Acoustic Devices AudioMoth v1.2', 384000, 16,
         'Grabadora autonoma; util para monitoreo pasivo.'),
        ('Zoom H5', 'Zoom H5 Handy Recorder', 96000, 24,
         'Grabadora portatil para campo.'),
        ('Sony PCM-A10', 'Sony PCM-A10', 192000, 24,
         'Grabadora portatil compacta.'),
        ('Raspberry Pi INMP441', 'MEMS INMP441', 48000, 32,
         'Nodo IoT economico para prototipos.')
)
INSERT INTO recording_device (name, model, sample_rate, bit_depth, notes)
SELECT d.name, d.model, d.sample_rate, d.bit_depth, d.notes
FROM device_data d
WHERE NOT EXISTS (
    SELECT 1 FROM recording_device rd WHERE rd.name = d.name
);

-- ── Sitio de desarrollo ─────────────────────────────────────────────────────
INSERT INTO recording_site (
    name, country, state_province, locality,
    latitude, longitude, altitude_m, habitat_type, notes
)
SELECT
    'Mexico -- Sitio de desarrollo',
    'Mexico', 'Sin especificar', 'Dataset publico y grabaciones de prueba',
    19.432608, -99.133209, 2240,
    'mixed',
    'Sitio semilla para pruebas; coordenadas de referencia en Ciudad de Mexico.'
WHERE NOT EXISTS (
    SELECT 1 FROM recording_site WHERE name = 'Mexico -- Sitio de desarrollo'
);

-- ── Dataset inicial ─────────────────────────────────────────────────────────
INSERT INTO dataset (name, version, description, split_type)
VALUES
    ('Mexico-Birds-v1', '0.1.0',
     'MVP de aves de Mexico. Fuentes iniciales: Xeno-canto filtrado por cnt:Mexico; iNaturalist opcional con place_id.',
     'stratified')
ON CONFLICT (name) DO UPDATE
SET version = EXCLUDED.version,
    description = EXCLUDED.description,
    split_type = EXCLUDED.split_type;

-- ── Taxonomia: ordenes ──────────────────────────────────────────────────────
WITH aves AS (
    SELECT id FROM taxa_class WHERE name = 'Aves'
),
order_data(name, common_name) AS (
    VALUES
        ('Passeriformes', 'Paseriformes'),
        ('Piciformes', 'Carpinteros y afines'),
        ('Columbiformes', 'Palomas y tortolas'),
        ('Galliformes', 'Galliformes'),
        ('Cuculiformes', 'Cucos y correcaminos'),
        ('Coraciiformes', 'Momotos y afines'),
        ('Strigiformes', 'Buhos y tecolotes')
)
INSERT INTO taxa_order (class_id, name, common_name)
SELECT aves.id, od.name, od.common_name
FROM aves CROSS JOIN order_data od
ON CONFLICT (class_id, name) DO UPDATE
SET common_name = EXCLUDED.common_name;

-- ── Taxonomia: familias ─────────────────────────────────────────────────────
WITH family_data(order_name, family_name, common_name) AS (
    VALUES
        ('Passeriformes', 'Icteridae', 'Tordos y bolseros'),
        ('Passeriformes', 'Turdidae', 'Zorzales'),
        ('Passeriformes', 'Tyrannidae', 'Tiranidos'),
        ('Passeriformes', 'Troglodytidae', 'Chivirines'),
        ('Passeriformes', 'Mimidae', 'Cuitlacoches'),
        ('Passeriformes', 'Fringillidae', 'Pinzones'),
        ('Passeriformes', 'Parulidae', 'Chipes'),
        ('Passeriformes', 'Vireonidae', 'Vireos'),
        ('Passeriformes', 'Corvidae', 'Cuervos y charas'),
        ('Piciformes', 'Picidae', 'Carpinteros'),
        ('Columbiformes', 'Columbidae', 'Palomas y tortolas'),
        ('Galliformes', 'Cracidae', 'Chachalacas'),
        ('Cuculiformes', 'Cuculidae', 'Cucos y correcaminos'),
        ('Coraciiformes', 'Momotidae', 'Momotos'),
        ('Strigiformes', 'Strigidae', 'Buhos y tecolotes')
)
INSERT INTO taxa_family (order_id, name, common_name)
SELECT o.id, fd.family_name, fd.common_name
FROM family_data fd
JOIN taxa_order o ON o.name = fd.order_name
JOIN taxa_class c ON c.id = o.class_id AND c.name = 'Aves'
ON CONFLICT (order_id, name) DO UPDATE
SET common_name = EXCLUDED.common_name;

-- ── Taxonomia: generos ──────────────────────────────────────────────────────
WITH genus_data(family_name, genus_name) AS (
    VALUES
        ('Icteridae', 'Quiscalus'),
        ('Icteridae', 'Icterus'),
        ('Turdidae', 'Turdus'),
        ('Tyrannidae', 'Pitangus'),
        ('Tyrannidae', 'Myiozetetes'),
        ('Picidae', 'Melanerpes'),
        ('Troglodytidae', 'Campylorhynchus'),
        ('Troglodytidae', 'Thryophilus'),
        ('Mimidae', 'Toxostoma'),
        ('Columbidae', 'Zenaida'),
        ('Columbidae', 'Columbina'),
        ('Cracidae', 'Ortalis'),
        ('Cuculidae', 'Crotophaga'),
        ('Cuculidae', 'Geococcyx'),
        ('Momotidae', 'Momotus'),
        ('Strigidae', 'Glaucidium'),
        ('Fringillidae', 'Haemorhous'),
        ('Parulidae', 'Setophaga'),
        ('Vireonidae', 'Vireo'),
        ('Corvidae', 'Cyanocorax')
)
INSERT INTO taxa_genus (family_id, name)
SELECT f.id, gd.genus_name
FROM genus_data gd
JOIN taxa_family f ON f.name = gd.family_name
JOIN taxa_order o ON o.id = f.order_id
JOIN taxa_class c ON c.id = o.class_id AND c.name = 'Aves'
ON CONFLICT (family_id, name) DO NOTHING;

-- ── Especies objetivo: MVP aves de Mexico ───────────────────────────────────
WITH species_data(
    genus_name, epithet, common_name_es, common_name_en,
    freq_min_hz, freq_max_hz, freq_dom_hz, notes
) AS (
    VALUES
        ('Quiscalus', 'mexicanus', 'Zanate mexicano', 'Great-tailed Grackle',
         500, 8000, 2500, 'Urbana y comun; buena clase inicial.'),
        ('Turdus', 'grayi', 'Mirlo pardo', 'Clay-colored Thrush',
         1000, 8000, 3000, 'Vocal y frecuente en el sureste.'),
        ('Pitangus', 'sulphuratus', 'Luis bienteveo', 'Great Kiskadee',
         500, 6000, 2000, 'Llamada fuerte y distintiva.'),
        ('Myiozetetes', 'similis', 'Luis gregario', 'Social Flycatcher',
         800, 7000, 2500, 'Tiranido comun en zonas tropicales.'),
        ('Melanerpes', 'aurifrons', 'Carpintero frente dorada', 'Golden-fronted Woodpecker',
         500, 7000, 2000, 'Tamborileo y llamadas audibles.'),
        ('Campylorhynchus', 'brunneicapillus', 'Matraca del desierto', 'Cactus Wren',
         800, 9000, 3500, 'Comun en matorral arido del norte.'),
        ('Thryophilus', 'sinaloa', 'Chivirin sinaloense', 'Sinaloa Wren',
         1000, 9000, 3500, 'Especie vocal del occidente de Mexico.'),
        ('Icterus', 'pustulatus', 'Bolsero dorso rayado', 'Streak-backed Oriole',
         1000, 9000, 4000, 'Canto claro en ambientes abiertos.'),
        ('Toxostoma', 'curvirostre', 'Cuitlacoche pico curvo', 'Curve-billed Thrasher',
         800, 8000, 3000, 'Comun en zonas aridas y urbanas.'),
        ('Zenaida', 'asiatica', 'Paloma ala blanca', 'White-winged Dove',
         300, 3000, 800, 'Arrullo grave, util como clase contrastante.'),
        ('Columbina', 'inca', 'Tortolita cola larga', 'Inca Dove',
         400, 4000, 1200, 'Arrullo repetitivo en ambientes urbanos y secos.'),
        ('Ortalis', 'vetula', 'Chachalaca oriental', 'Plain Chachalaca',
         300, 3000, 900, 'Coro fuerte y muy distintivo.'),
        ('Crotophaga', 'sulcirostris', 'Garrapatero pijuy', 'Groove-billed Ani',
         300, 5000, 1200, 'Llamadas nasales distintivas.'),
        ('Momotus', 'lessonii', 'Momoto corona azul', 'Lesson''s Motmot',
         400, 5000, 1200, 'Vocalizacion baja de bosque tropical.'),
        ('Glaucidium', 'brasilianum', 'Tecolote bajeño', 'Ferruginous Pygmy-Owl',
         500, 8000, 2000, 'Silbidos repetitivos, buen objetivo nocturno.'),
        ('Geococcyx', 'californianus', 'Correcaminos norteño', 'Greater Roadrunner',
         300, 3000, 800, 'Vocalizacion grave en zonas aridas.'),
        ('Haemorhous', 'mexicanus', 'Pinzon mexicano', 'House Finch',
         1000, 9000, 4000, 'Comun y vocal en zonas urbanas.'),
        ('Setophaga', 'petechia', 'Chipe amarillo', 'Yellow Warbler',
         2000, 10000, 5000, 'Canto agudo; util para bandas altas.'),
        ('Vireo', 'hypochryseus', 'Vireo dorado', 'Golden Vireo',
         1500, 9000, 4000, 'Vireo mexicano vocal.'),
        ('Cyanocorax', 'yncas', 'Chara verde', 'Green Jay',
         500, 7000, 1800, 'Llamadas sociales variadas y fuertes.')
)
INSERT INTO species (
    genus_id, epithet, common_name_es, common_name_en,
    iucn_status, acoustic_group,
    freq_min_hz, freq_max_hz, freq_dom_hz, notes
)
SELECT
    g.id,
    sd.epithet,
    sd.common_name_es,
    sd.common_name_en,
    'LC',
    'bird',
    sd.freq_min_hz,
    sd.freq_max_hz,
    sd.freq_dom_hz,
    sd.notes
FROM species_data sd
JOIN taxa_genus g ON g.name = sd.genus_name
JOIN taxa_family f ON f.id = g.family_id
JOIN taxa_order o ON o.id = f.order_id
JOIN taxa_class c ON c.id = o.class_id AND c.name = 'Aves'
ON CONFLICT (genus_id, epithet) DO UPDATE
SET common_name_es = EXCLUDED.common_name_es,
    common_name_en = EXCLUDED.common_name_en,
    iucn_status = EXCLUDED.iucn_status,
    acoustic_group = EXCLUDED.acoustic_group,
    freq_min_hz = EXCLUDED.freq_min_hz,
    freq_max_hz = EXCLUDED.freq_max_hz,
    freq_dom_hz = EXCLUDED.freq_dom_hz,
    notes = EXCLUDED.notes;

DO $$
DECLARE
    n_species INT;
BEGIN
    SELECT COUNT(*) INTO n_species
    FROM species_full
    WHERE acoustic_group = 'bird';
    RAISE NOTICE 'Seed completado: % especies de aves registradas.', n_species;
END $$;
