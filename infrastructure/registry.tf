locals {
  parent_dir = abspath("${path.module}/..")
  registry_name  = "${var.name_prefix}-registry"
  repo_name      = "${var.name_prefix}-worker"
  image_name = "onnx"
}


data "yandex_client_config" "client" {}

resource "yandex_container_registry" "cr" {
  name = local.registry_name

}


resource "yandex_container_repository" "repo" {
	name = "${yandex_container_registry.cr.id}/${local.repo_name}"
}



resource "yandex_container_repository_lifecycle_policy" "lp" {
  name          = "${var.name_prefix}-lifecycle-policy-name"
  status        = "active"
  repository_id = yandex_container_repository.repo.id

  rule {
    description  = "Spare 1"
    untagged     = true
    tag_regexp   = ".*"
    retained_top = 1
  }
}








#resource "docker_image" "worker" {
#  name = "cr.yandex/${yandex_container_registry.cr.id}/worker:latest"
#
#  build {
#    context    = "../src/"
#    dockerfile = "Dockerfile.worker"
#  }
#
#  lifecycle {
#    replace_triggered_by = [
#      yandex_container_registry.cr.id
#    ]
#  }
#
#  depends_on = [yandex_container_registry.cr]
#}
#
#
#
#resource "docker_registry_image" "worker_push" {
#  name = docker_image.worker.name
#
#  keep_remotely = false
#
#
#  lifecycle {
#    replace_triggered_by = [
#      yandex_container_registry.cr.id
#    ]
#  }
#}
#
#
#
#
#


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
