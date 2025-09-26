CREATE TABLE IF NOT EXISTS airports (
  airport_id   INT PRIMARY KEY,
  name         VARCHAR(255),
  city         VARCHAR(255),
  country      VARCHAR(255),
  iata         VARCHAR(10),
  icao         VARCHAR(10),
  latitude     DOUBLE,
  longitude    DOUBLE,
  altitude     INT,
  timezone     VARCHAR(50),
  dst          VARCHAR(10),
  tz           VARCHAR(64),
  type         VARCHAR(50),
  source       VARCHAR(50)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS airlines (
  airline_id INT PRIMARY KEY,
  name       VARCHAR(255),
  alias      VARCHAR(255),
  iata       VARCHAR(10),
  icao       VARCHAR(10),
  callsign   VARCHAR(255),
  country    VARCHAR(255),
  active     VARCHAR(5)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS routes (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  airline       VARCHAR(10),
  airline_id    INT,
  src           VARCHAR(10),
  src_id        INT,
  dst           VARCHAR(10),
  dst_id        INT,
  codeshare     VARCHAR(5),
  stops         INT,
  equipment     VARCHAR(64),
  route_key     VARCHAR(191)
                 GENERATED ALWAYS AS (
                   CONCAT_WS(
                     '|',
                     UPPER(COALESCE(airline, '')),
                     UPPER(COALESCE(src, '')),
                     UPPER(COALESCE(dst, '')),
                     COALESCE(codeshare, ''),
                     COALESCE(CAST(stops AS CHAR), ''),
                     COALESCE(equipment, '')
                   )
                 ) STORED,
  UNIQUE KEY uniq_route_key (route_key)
) ENGINE=InnoDB;
