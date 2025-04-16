# Vector Embeeding

## Setup
- Clone the repository and run the setup script
```bash
git clone https://github.com/itsemon245/image-vector-embedding.git && cd image-vector-embedding && ./setup
```
- Start the docker container (you might want to change the port in the env)
```bash
docker-compose up -d
```
### Install pgvector extension in postgres
#### If you are not using dockre
```bash
./utils/pgv-init
```

#### If you are using docker
- Enter in your container shell (change the container name)
```bash
docker exec -it my_container bash
```
- Set your database name, user and password
```bash
export DB=my_database
export USER=my_user
export PORT=5432
export HOST=localhost
```
- Copy and paste the script in terminal(you might be prompted to enter password)
```bash
# Install required dependencies
apt-get update
apt-get install -y build-essential postgresql-server-dev-${PG_MAJOR:-17} libpq-dev

# Clone the pgvector repository
git clone https://github.com/pgvector/pgvector.git
cd pgvector
#install
make
make install

# Enable the extension in the database(you might be prompted to enter password)
psql -d ${DB} -U ${USER} -h ${HOST} -p ${PORT} -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
>![NOTE]
>You can use the docker-compose file in the misc folder to start a postgres container


## Usage
### Embed an image
- To embed an image, use the `/embed` endpoint
```bash
curl --location 'http://localhost:8787/embed' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <your-app-key>' \
--data '[
    "publicly-accessible-image-url"
]'
```

> [!IMPORTANT]
> **Use 512 dimensional vector for clip-vit-base-patch32**

### Search query to find similar images(Postgres only)
```sql
WITH ranked AS (
  SELECT *, embedding <=> '[...]' AS distance -- replace with embedding with your vector column
  FROM images -- replace with your table name
)
SELECT *
FROM ranked
WHERE distance < 0.099 --the higher lower the value the more similar the image is, increase the value to get more images but less similar
ORDER BY distance
LIMIT 2;
```
