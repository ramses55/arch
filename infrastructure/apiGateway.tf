locals {
	apiGateway_name = "${var.name_prefix}-gateway"
	sa_g_name = "${var.name_prefix}-sa-for-gateway"
	file_g_path =  "../src/yandex-cloud/api-gateway"
}



resource "yandex_iam_service_account" "sa-g" {
	name = local.sa_g_name	
}


#TODO: make for_each
resource "yandex_resourcemanager_folder_iam_member" "gf" {
	folder_id = var.folder_id
	role = "functions.functionInvoker"
	member = "serviceAccount:${yandex_iam_service_account.sa-g.id}"
}



resource "yandex_resourcemanager_folder_iam_member" "gw" {
	folder_id = var.folder_id
	role = "serverless.workflows.executor"
	member = "serviceAccount:${yandex_iam_service_account.sa-g.id}"
}




resource "yandex_api_gateway" "gateway"{
	depends_on = [yandex_serverless_workflow.workflow]
	name = local.apiGateway_name
	spec = templatefile( local.file_g_path, {
		sa-g-id = yandex_iam_service_account.sa-g.id,
		workflow-id = yandex_serverless_workflow.workflow.id,
		num-mes-id = yandex_function.num_mes.id,
		workflow_url = yandex_serverless_workflow.workflow.execution_url
	})
}
