variable "aws_region" { type = string default = "us-east-1" }
variable "ecr_repository" { type = string default = "ai-research-agent" }
variable "image_tag" { type = string default = "latest" }
variable "desired_count" { type = number default = 2 }
variable "gemini_api_key" { type = string sensitive = true }
variable "chroma_host" { type = string }
variable "chroma_port" { type = number default = 8000 }
variable "neo4j_uri" { type = string }
variable "neo4j_username" { type = string default = "neo4j" }
variable "neo4j_password" { type = string sensitive = true }
