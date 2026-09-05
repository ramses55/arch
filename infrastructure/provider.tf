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

    docker = {
          source  = "kreuzwerker/docker"
          version = "~> 4.5"
    }
  }

  required_version = ">= 1.00"
}

provider "yandex" {
}


provider "docker" {
	registry_auth {
	    address  = "cr.yandex"
	    username = "iam"
	    password = data.yandex_client_config.client.iam_token
	  }
}




provider "random" {
}

provider "archive" {
}
