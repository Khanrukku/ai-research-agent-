output "ecr_repository_url" { value = aws_ecr_repository.agent.repository_url }
output "ecs_cluster" { value = aws_ecs_cluster.agent.name }
output "s3_reports_bucket" { value = aws_s3_bucket.reports.bucket }
output "log_group" { value = aws_cloudwatch_log_group.agent.name }
