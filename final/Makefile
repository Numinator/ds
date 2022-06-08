.PHONY: database

database: psql.cmd
	sudo -u postgres psql -E -c "DROP DATABASE IF EXISTS marche" \
-c "DROP USER IF EXISTS ghibellini" \
-c "CREATE DATABASE marche" \
-c "CREATE USER ghibellini WITH PASSWORD 'guelfi'" \
-c "GRANT ALL PRIVILEGES ON DATABASE marche to ghibellini" \
-c "DROP TABLE IF EXISTS fnc_article CASCADE;DROP TABLE IF EXISTS author CASCADE;DROP TABLE IF EXISTS tags CASCADE;DROP TABLE IF EXISTS keywords CASCADE;DROP TABLE IF EXISTS metakeywords CASCADE;DROP TABLE IF EXISTS wn_article CASCADE;DROP TABLE IF EXISTS sources CASCADE;DROP TABLE IF EXISTS categories CASCADE; DROP VIEW IF EXISTS unified_article; DROP VIEW IF EXISTS unified_keywords;DROP VIEW IF EXISTS unified_sources" 


	sudo -u postgres psql -E \
-c "CREATE TABLE fnc_article ( id INT,  domain VARCHAR,  type VARCHAR,  url VARCHAR,  content VARCHAR,  scraped_at TIMESTAMP,  inserted_at TIMESTAMP, updated_at TIMESTAMP,  title VARCHAR (256),  meta_description VARCHAR,  summary VARCHAR, PRIMARY KEY (id));" \
-c "CREATE TABLE tags (a_id INT, tag VARCHAR, PRIMARY KEY (a_id, tag),FOREIGN KEY (a_id)    REFERENCES fnc_article (id)    ON UPDATE CASCADE ON DELETE CASCADE);" \
-c "CREATE TABLE keywords (a_id INT, keyword VARCHAR,PRIMARY KEY (a_id, keyword),FOREIGN KEY (a_id)    REFERENCES fnc_article (id)    ON UPDATE CASCADE ON DELETE CASCADE);" \
-c "CREATE TABLE metakeywords (a_id INT, mkeyword VARCHAR,PRIMARY KEY (a_id, mkeyword),FOREIGN KEY (a_id) REFERENCES fnc_article (id) ON UPDATE CASCADE ON DELETE CASCADE);" \
-c "CREATE TABLE author (a_id INT,author VARCHAR,PRIMARY KEY (a_id, author),FOREIGN KEY (a_id)    REFERENCES fnc_article (id) ON UPDATE CASCADE ON DELETE CASCADE);"

	sudo -u postgres psql -E \
-c "CREATE TABLE wn_article (id INT, content VARCHAR, publish_date TIMESTAMP, modified_date TIMESTAMP,title VARCHAR (256),PRIMARY KEY (id));" \
-c "CREATE TABLE sources (a_id INT,sources VARCHAR,PRIMARY KEY (a_id, sources),FOREIGN KEY (a_id) REFERENCES wn_article (id) ON UPDATE CASCADE ON DELETE CASCADE);" \
-c "CREATE TABLE categories (a_id INT, categories VARCHAR, PRIMARY KEY (a_id, categories),FOREIGN KEY (a_id) REFERENCES wn_article (id) ON UPDATE CASCADE ON DELETE CASCADE);"


	sudo -u postgres psql -E \
-c "CREATE VIEW unified_article AS (SELECT 'fnc', id, title, content, inserted_at as published, updated_at as updated FROM fnc_article) UNION (SELECT 'wn', id, title, content, publish_date as published, modified_date as updated FROM wn_article)" \
-c "CREATE VIEW unified_keywords AS (SELECT 'fnc', a_id, keyword FROM keywords) UNION (SELECT 'wn', a_id, categories as keyword FROM categories)" \
-c "CREATE VIEW unified_sources AS (SELECT 'fnc', a_id, author as sources FROM author) UNION (SELECT 'wn', a_id, sources FROM sources)"
