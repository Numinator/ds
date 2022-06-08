.PHONY: database

database: psql.cmd
	sudo -u postgres psql -E -c "DROP DATABASE IF EXISTS marche" \
	 -c "DROP USER IF EXISTS ghibellini" \
	 -c "CREATE DATABASE marche" \
	 -c "CREATE USER ghibellini WITH PASSWORD 'guelfi'" \
	 -c "GRANT ALL PRIVILEGES ON DATABASE marche to ghibellini"

