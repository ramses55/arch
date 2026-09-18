locals {
  ymq_name      = "${var.name_prefix}-ymq"
  ymq_dead_name = "${var.name_prefix}-ymq-dead"
  f_num_name    = "${var.name_prefix}-f-num-mes"
  f_push_name   = "${var.name_prefix}-f-push"
  f_num_path    = "../src/yandex-cloud/cloud-functions/num-mes/"
  f_push_path   = "../src/yandex-cloud/cloud-functions/push-to-queue/"
}


resource "yandex_message_queue" "dead_queue" {
  name       = local.ymq_dead_name
  fifo_queue = false

  access_key = var.yc_access_key
  secret_key = var.yc_secret_key

}


resource "yandex_message_queue" "main_queue" {
  name       = local.ymq_name
  fifo_queue = false


  access_key = var.yc_access_key
  secret_key = var.yc_secret_key

  redrive_policy = jsonencode({
    deadLetterTargetArn = yandex_message_queue.dead_queue.arn
    maxReceiveCount     = 3
  })
}




#service account for message the main message queue
resource "yandex_iam_service_account" "sa-mes" {
  name = "sa-num-mes"
}

resource "yandex_resourcemanager_folder_iam_member" "rq" {
  folder_id = var.folder_id
  role      = "ymq.reader"
  member    = "serviceAccount:${yandex_iam_service_account.sa-mes.id}"
}


resource "yandex_resourcemanager_folder_iam_member" "rw" {
  folder_id = var.folder_id
  role      = "ymq.writer"
  member    = "serviceAccount:${yandex_iam_service_account.sa-mes.id}"
}


resource "yandex_resourcemanager_folder_iam_member" "rs1" {
  folder_id = var.folder_id
  role      = "lockbox.payloadViewer"
  member    = "serviceAccount:${yandex_iam_service_account.sa-mes.id}"
}

resource "yandex_iam_service_account_static_access_key" "key-mes" {
  service_account_id = yandex_iam_service_account.sa-mes.id
  description        = "Static access key for num-mes function to read YMQ"

  output_to_lockbox {
    secret_id            = yandex_lockbox_secret.sb.id
    entry_for_access_key = "${local.f_num_name}-access_key"
    entry_for_secret_key = "${local.f_num_name}-secret_key"
  }
}



resource "yandex_lockbox_secret" "sb" {
  name = "${var.name_prefix}-secrets"
}

resource "yandex_lockbox_secret_version_hashed" "secrets_payload" {
  secret_id = yandex_lockbox_secret.sb.id

  key_1        = "oauth_token"
  text_value_1 = var.oauth_token
}


data "archive_file" "mes_zip" {
  type        = "zip"
  source_dir  = local.f_num_path
  output_path = "${path.module}/num-mes.zip"
}

resource "yandex_function" "num_mes" {
  name               = local.f_num_name
  description        = "Returns json containig number of messages in queue"
  service_account_id = yandex_iam_service_account.sa-mes.id

  user_hash         = "v1"
  runtime           = "python314"
  entrypoint        = "num-mes.handler"
  memory            = "128"
  execution_timeout = "10"


  content {
    zip_filename = data.archive_file.mes_zip.output_path
  }


  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-mes.output_to_lockbox_version_id
    key                  = "${local.f_num_name}-access_key"
    environment_variable = "access_key_id"
  }

  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-mes.output_to_lockbox_version_id
    key                  = "${local.f_num_name}-secret_key"
    environment_variable = "access_key"
  }

  environment = {
    queue_url = yandex_message_queue.main_queue.id
  }

depends_on = [yandex_resourcemanager_folder_iam_member.rs1]

}



data "archive_file" "push_zip" {
  type        = "zip"
  source_dir  = local.f_push_path
  output_path = "${path.module}/push-to-queue.zip"
}

resource "yandex_function" "push-to-queue" {
  name               = local.f_push_name
  description        = "Returns json containig number of messages it pushed to the queue"
  service_account_id = yandex_iam_service_account.sa-mes.id

  user_hash         = data.archive_file.push_zip.output_base64sha256
  runtime           = "python314"
  entrypoint        = "push-to-queue.handler"
  memory            = "256"
  execution_timeout = "10"


  content {
    zip_filename = data.archive_file.push_zip.output_path
  }


  #TODO: make dynamic block
  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-mes.output_to_lockbox_version_id
    key                  = "${local.f_num_name}-access_key"
    environment_variable = "access_key_id"
  }

  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_iam_service_account_static_access_key.key-mes.output_to_lockbox_version_id
    key                  = "${local.f_num_name}-secret_key"
    environment_variable = "access_key"
  }

  secrets {
    id                   = yandex_lockbox_secret.sb.id
    version_id           = yandex_lockbox_secret_version_hashed.secrets_payload.id
    key                  = "oauth_token"
    environment_variable = "oauth_token"
  }
  environment = {
    queue_url = yandex_message_queue.main_queue.id
  }

	depends_on = [yandex_resourcemanager_folder_iam_member.rs1]
}

