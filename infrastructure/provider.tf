terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
    }

    random = {
      source = "hashicorp/random"
    }

    archive = {
      source = "hashicorp/archive"
    }
  }

  required_version = ">= 1.00"
}

provider "yandex" {
}

provider "random" {
}

provider "archive" {
}
