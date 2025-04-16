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
- If your database is in you host machine run the script(it will ask for the database name, user and password)
```bash
./utils/pgv-init
```

- If your database is in a docker container run this from this directory(change the container name and database credentials accordingly)
```bash
docker exec -i my_container bash -s -- dbname dbuser dbpassword < ./utils/pgv-init
```


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
