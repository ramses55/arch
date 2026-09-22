locals {
  container_name = "${var.name_prefix}-container"
  sa_name        = "${var.name_prefix}-sa-con"
  bucket_name    = "${var.name_prefix}-bucket"
}



variable "container_image" {
	type = string

	default = "cr.yandex/mirror/library/alpine:latest"
}




#Container will need service account with following roles
variable "container_roles" {
  type = set(string)

  default = [
 	"container-registry.images.puller",
 	"ymq.reader",
 	"storage.editor",
 	"lockbox.payloadViewer",
 	"ai.vision.user"
  ]
}

resource "yandex_iam_service_account" "sa-con" {
  name        = local.sa_name
  description = "Service account for Serverless Container"
}


resource "yandex_resourcemanager_folder_iam_member" "cr" {
  for_each = var.container_roles


  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.sa-con.id}"
}


resource "yandex_storage_bucket" "b" {
  folder_id     = var.folder_id
  bucket_prefix = local.bucket_name
  force_destroy = true

  max_size = 1073741824 # 1 GB
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


resource "yandex_lockbox_secret_version_hashed" "ocr_payload" {
  secret_id = yandex_lockbox_secret.sb.id

  key_1        = "${local.container_name}-ocr_key"
  text_value_1 = yandex_iam_service_account_api_key.ocr_api_key.secret_key
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
    #generic image to create container, actual image will be pushed manually and then container will be updated with terraform apply -var...
    url = var.container_image

    environment = {
      queue_url   = yandex_message_queue.main_queue.id
      res_limit   = 2
      img_limit   = 30
      folder_id   = var.folder_id
      bucket_name = yandex_storage_bucket.b.bucket
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


  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_lockbox_secret_version_hashed.ocr_payload.id
    key                  ="${local.container_name}-ocr_key"
    environment_variable = "api_key"
  }


  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_lockbox_secret_version_hashed.secrets_payload.id
    key                  = "oauth_token"
    environment_variable = "oauth_token"
  }

}
