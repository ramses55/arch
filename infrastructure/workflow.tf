locals {
  workflow_name = "${var.name_prefix}-workflow"
  workflow_path = "../src/yandex-cloud/workflow"
}



#Workflow will need service account with following roles
variable "workflow_roles" {
  type = set(string)

  default = [
    "functions.functionInvoker",
    "serverless-containers.containerInvoker"
  ]
}

resource "yandex_iam_service_account" "sa-w" {
  name = "${var.name_prefix}-sa-workflow"
}

resource "yandex_resourcemanager_folder_iam_member" "wr" {
  for_each = var.workflow_roles

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.sa-w.id}"
}




resource "yandex_serverless_workflow" "workflow" {
  depends_on         = [yandex_serverless_container.container, yandex_function.push-to-queue, yandex_function.num_mes]
  name               = local.workflow_name
  folder_id          = var.folder_id
  service_account_id = yandex_iam_service_account.sa-w.id

  specification = {
    spec_yaml = templatefile(local.workflow_path, {
      num-mes-id       = yandex_function.num_mes.id,
      push-to-queue-id = yandex_function.push-to-queue.id,
      container-id     = yandex_serverless_container.container.id
    })
  }

}

