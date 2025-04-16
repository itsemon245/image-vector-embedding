# Vector Embeeding

## Setup

```bash
git clone https://github.com/emon/image-vector-embedding.git
cd image-vector-embedding
docker-compose up -d
```

## Usage
### Embed an image
- To embed an image, use the `/embed` endpoint
```bash
curl --location 'http://localhost:8787/embed' \
--header 'Content-Type: application/json' \
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
