terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" { filter { name = "vpc-id" values = [data.aws_vpc.default.id] } }

data "aws_iam_policy_document" "ecs_assume" {
  statement { actions = ["sts:AssumeRole"] principals { type = "Service" identifiers = ["ecs-tasks.amazonaws.com"] } }
}

resource "aws_ecr_repository" "agent" {
  name                 = var.ecr_repository
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_cloudwatch_log_group" "agent" {
  name              = "/ecs/${var.ecr_repository}"
  retention_in_days = 14
}

resource "aws_iam_role" "task_execution" {
  name               = "${var.ecr_repository}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"]
}

resource "aws_iam_role" "task" {
  name               = "${var.ecr_repository}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  inline_policy {
    name = "research-artifacts"
    policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], Resource = [aws_s3_bucket.reports.arn, "${aws_s3_bucket.reports.arn}/*"] }] })
  }
}

resource "aws_s3_bucket" "reports" {
  bucket_prefix = "${var.ecr_repository}-reports-"
  force_destroy = true
}

resource "aws_ecs_cluster" "agent" { name = "${var.ecr_repository}-cluster" }

resource "aws_ecs_task_definition" "agent" {
  family                   = var.ecr_repository
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name      = "api"
    image     = "${aws_ecr_repository.agent.repository_url}:${var.image_tag}"
    essential = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "APP_ENV", value = "production" },
      { name = "CHROMA_HOST", value = var.chroma_host },
      { name = "CHROMA_PORT", value = tostring(var.chroma_port) },
      { name = "NEO4J_URI", value = var.neo4j_uri },
      { name = "NEO4J_USERNAME", value = var.neo4j_username },
      { name = "NEO4J_PASSWORD", value = var.neo4j_password },
      { name = "GEMINI_API_KEY", value = var.gemini_api_key }
    ]
    logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.agent.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "api" } }
  }])
}

resource "aws_security_group" "agent" {
  name   = "${var.ecr_repository}-sg"
  vpc_id = data.aws_vpc.default.id
  ingress { from_port = 8000 to_port = 8000 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  egress { from_port = 0 to_port = 0 protocol = "-1" cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_ecs_service" "agent" {
  name            = var.ecr_repository
  cluster         = aws_ecs_cluster.agent.id
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.agent.id]
    assign_public_ip = true
  }
}
