CREATE TABLE IF NOT EXISTS airports_emb (
  airport_id INT PRIMARY KEY,
  desc_text  TEXT,
  emb        VECTOR(768) NOT NULL,
  FOREIGN KEY (airport_id) REFERENCES airports(airport_id)
);
CREATE VECTOR INDEX airports_emb_idx ON airports_emb (emb);

CREATE TABLE IF NOT EXISTS airlines_emb (
  airline_id INT PRIMARY KEY,
  desc_text  TEXT,
  emb        VECTOR(768) NOT NULL,
  FOREIGN KEY (airline_id) REFERENCES airlines(airline_id)
);
CREATE VECTOR INDEX airlines_emb_idx ON airlines_emb (emb);

CREATE TABLE IF NOT EXISTS routes_emb (
  route_id BIGINT PRIMARY KEY,
  desc_text TEXT,
  emb      VECTOR(768) NOT NULL,
  FOREIGN KEY (route_id) REFERENCES routes(id)
);
CREATE VECTOR INDEX routes_emb_idx ON routes_emb (emb);