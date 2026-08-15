# AWS deployment

This Terraform stack deploys the FastAPI research workers to **ECS Fargate** behind a reusable task definition, creates an **ECR** image repository, **CloudWatch** logs, and an **S3** bucket for research artifacts. `desired_count=2` demonstrates stateless horizontal worker scaling.

ChromaDB and Neo4j are intentionally externalized so the same worker image can point at managed/self-hosted data services without embedding state in the task. For production, store API credentials in AWS Secrets Manager rather than Terraform variables.

```bash
terraform init
terraform plan -var='gemini_api_key=...' \
  -var='chroma_host=your-chroma-host' \
  -var='neo4j_uri=bolt+s://your-neo4j' \
  -var='neo4j_password=...'
terraform apply
```

Build and push the image to the ECR URL printed by Terraform, then update `image_tag` and apply again.
