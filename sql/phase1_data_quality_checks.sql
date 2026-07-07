
-- ============================================================
-- Cookie Cats A/B Test — SQL Data Quality Queries
-- Phase 1: Data Understanding & Cleaning
-- ============================================================

-- Query 1: Total row count
SELECT COUNT(*) AS total_rows 
FROM cookie_cats;

-- ────────────────────────────────────────────────────────────

-- Query 2: Duplicate userid check
SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT userid) AS unique_users,
    COUNT(*) - COUNT(DISTINCT userid) AS duplicate_count
FROM cookie_cats;

-- ────────────────────────────────────────────────────────────

-- Query 3: Null check per column
SELECT
    COUNT(*) - COUNT(userid)            AS userid_nulls,
    COUNT(*) - COUNT(version)           AS version_nulls,
    COUNT(*) - COUNT(sum_gamerounds)    AS sum_gamerounds_nulls,
    COUNT(*) - COUNT(retention_1)       AS retention_1_nulls,
    COUNT(*) - COUNT(retention_7)       AS retention_7_nulls
FROM cookie_cats;

-- ────────────────────────────────────────────────────────────

-- Query 4: Group sizes and balance
SELECT
    version,
    COUNT(*) AS player_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM cookie_cats), 2) AS percentage
FROM cookie_cats
GROUP BY version
ORDER BY version;

-- ────────────────────────────────────────────────────────────

-- Query 5: Aggregate stats per group
SELECT
    version,
    COUNT(*)                                    AS player_count,
    ROUND(AVG(sum_gamerounds), 2)               AS avg_gamerounds,
    ROUND(AVG(CAST(retention_1 AS FLOAT)), 4)   AS retention_1_rate,
    ROUND(AVG(CAST(retention_7 AS FLOAT)), 4)   AS retention_7_rate
FROM cookie_cats
GROUP BY version
ORDER BY version;

-- ────────────────────────────────────────────────────────────

-- Query 6: Top 10 players by game rounds (outlier investigation)
SELECT 
    userid, 
    version, 
    sum_gamerounds 
FROM cookie_cats 
ORDER BY sum_gamerounds DESC
LIMIT 10;

-- ────────────────────────────────────────────────────────────

-- Query 7: Outlier quantification
SELECT
    MAX(sum_gamerounds)                                    AS max_value,
    ROUND(AVG(sum_gamerounds), 2)                          AS mean_value,
    COUNT(*) / 2                                           AS median_approx_n,
    (SELECT sum_gamerounds 
     FROM cookie_cats 
     ORDER BY sum_gamerounds 
     LIMIT 1 
     OFFSET (SELECT COUNT(*)/2 FROM cookie_cats))          AS median_approx
FROM cookie_cats;
