locals {
  apiGateway_name = "${var.name_prefix}-gateway"
  sa_g_name       = "${var.name_prefix}-sa-for-gateway"
  apiGateway_path = "../src/yandex-cloud/api-gateway"
}


#API Gateway will need service account with following roles
variable "apiGateway_roles" {
  type = set(string)

  default = [
    "functions.functionInvoker",
    "serverless.workflows.executor"
  ]
}



resource "yandex_iam_service_account" "sa-g" {
  name = local.sa_g_name
}




resource "yandex_resourcemanager_folder_iam_member" "gr" {
  for_each = var.apiGateway_roles

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.sa-g.id}"
}




resource "yandex_api_gateway" "gateway" {
  depends_on = [yandex_serverless_workflow.workflow]
  name       = local.apiGateway_name
  spec = templatefile(local.apiGateway_path, {
    sa-g-id      = yandex_iam_service_account.sa-g.id,
    workflow-id  = yandex_serverless_workflow.workflow.id,
    num-mes-id   = yandex_function.num_mes.id,
    workflow_url = yandex_serverless_workflow.workflow.execution_url
  })
}
