locals {
  parent_dir    = abspath("${path.module}/..")
  registry_name = "${var.name_prefix}-registry"
  repo_name     = "${var.name_prefix}-worker"
  image_name    = "onnx"
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

