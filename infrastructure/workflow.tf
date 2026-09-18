locals {
	workflow_name = "${var.name_prefix}-workflow"
	file_path = "../src/yandex-cloud/workflow"
}

resource "yandex_iam_service_account" "sa-w" {
	name = "${var.name_prefix}-sa-workflow"
}

#TODO: use for_each
resource "yandex_resourcemanager_folder_iam_member" "wf" {
	folder_id = var.folder_id
	role = "functions.functionInvoker"
  	member    = "serviceAccount:${yandex_iam_service_account.sa-w.id}"
}


resource "yandex_resourcemanager_folder_iam_member" "wc" {
	folder_id = var.folder_id
	role = "serverless-containers.containerInvoker"
  	member    = "serviceAccount:${yandex_iam_service_account.sa-w.id}"
}


resource "yandex_serverless_workflow" "workflow" {
	depends_on = [yandex_serverless_container.container, yandex_function.push-to-queue, yandex_function.num_mes ]
	name = local.workflow_name
	folder_id = var.folder_id
	service_account_id = yandex_iam_service_account.sa-w.id
	
	specification = {
		spec_yaml  = templatefile( local.file_path, {
			num-mes-id = yandex_function.num_mes.id,
			push-to-queue-id = yandex_function.push-to-queue.id,
			container-id = yandex_serverless_container.container.id
		})
	}
	
}

