locals {
  container_name = "${var.name_prefix}-container"
  registry_name  = "${var.name_prefix}-registry"
  repo_name      = "${var.name_prefix}-worker"
  sa_name 	 = "${var.name_prefix}-sa-con"
  bucket_name 	 = "${var.name_prefix}-bucket"

}

resource "yandex_iam_service_account" "sa-con" {
  name        = local.sa_name
  description = "Service account for Serverless Container"
}


#TODO: use for_each
resource "yandex_resourcemanager_folder_iam_member" "pull" {
  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}


resource "yandex_resourcemanager_folder_iam_member" "rq1" {
  folder_id = var.folder_id
  role      = "ymq.reader"
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}



resource "yandex_resourcemanager_folder_iam_member" "s3" {
  folder_id = var.folder_id
  role      = "storage.editor"
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}



resource "yandex_storage_bucket" "b" {
  folder_id = var.folder_id
  bucket_prefix = "${local.bucket_name}"
  force_destroy = true

  max_size = 1073741824
}


resource "yandex_resourcemanager_folder_iam_member" "rs2" {
  folder_id = var.folder_id
  role      = "lockbox.payloadViewer"
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "ocr" {
  folder_id = var.folder_id 
  role      = "ai.vision.user" 
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}


resource "yandex_iam_service_account_static_access_key" "key-con" {
  service_account_id = yandex_iam_service_account.sa-con.id
  description        = "Static access key for container to use S3 and SQS"

  output_to_lockbox {
    secret_id            = yandex_lockbox_secret.sb.id
    entry_for_access_key = "${local.container_name}-access_key"
    entry_for_secret_key = "${local.container_name}-secret_key"
  }
}

resource "yandex_iam_service_account_api_key" "ocr_api_key" {
  service_account_id = yandex_iam_service_account.sa-con.id
  description        = "API key for OCR"
  
  scopes = [
    "yc.ai.vision.execute"
  ]
}

resource "yandex_serverless_container" "container" {
  name               = local.container_name
  description        = "Worker"
  memory             = 3072
  execution_timeout  = "540s"
  cores              = 1
  core_fraction      = 100
  service_account_id = yandex_iam_service_account.sa-con.id


  runtime {
    type = "task"
  }


  image {
    url = docker_registry_image.worker_push.name

    environment = {
      queue_url = yandex_message_queue.main_queue.id
      res_limit = 2
      img_limit = 30
      folder_id = var.folder_id
      bucket_name = yandex_storage_bucket.b.bucket
      api_key = yandex_iam_service_account_api_key.ocr_api_key.secret_key
      oauth_token = var.oauth_token
    }
  }


  provision_policy {
    min_instances = 0
  }



  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-con.output_to_lockbox_version_id
    key                  = "${local.container_name}-access_key"
    environment_variable = "access_key_id"
  }

  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-con.output_to_lockbox_version_id
    key                  = "${local.container_name}-secret_key"
    environment_variable = "access_key"
  }

  depends_on = [ docker_registry_image.worker_push ]
}











##########################################

data "yandex_client_config" "client" {}

resource "yandex_container_registry" "cr" {
  name = local.registry_name

}


resource "yandex_container_repository" "repo" {
	name = "${yandex_container_registry.cr.id}/${local.repo_name}"

}


resource "docker_image" "worker" {
  name = "cr.yandex/${yandex_container_registry.cr.id}/worker:latest"

  build {
    context    = "../src/"
    dockerfile = "Dockerfile.worker"
  }

  lifecycle {
    replace_triggered_by = [
      yandex_container_registry.cr.id
    ]
  }

  depends_on = [yandex_container_registry.cr]
}



resource "docker_registry_image" "worker_push" {
  name = docker_image.worker.name

  keep_remotely = false


  lifecycle {
    replace_triggered_by = [
      yandex_container_registry.cr.id
    ]
  }
}







#resource "docker_registry_image" "worker" {
#  name = "cr.yandex/${yandex_container_repository.repo.name}:latest"
#
#  build {
#    context    = "../src/"
#    dockerfile = "Dockerfile.worker"
#  }
#  depends_on = [ yandex_container_repository.repo, yandex_container_registry.cr ]
#}
#


