locals {
  ymq_name      = "${var.name_prefix}-ymq"
  ymq_dead_name = "${var.name_prefix}-ymq-dead"
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
