variable "folder_id" {
  description = "(Optional) - Yandex Cloud Folder ID where resources will be created."
  type        = string
}



variable "name_prefix" {
  description = "(Optional) - Yandex Cloud Folder ID where resources will be created."
  type        = string
  default     = "test"
}



variable "yc_access_key" {
  type        = string
  description = "Static access key ID for the service account"
}

variable "yc_secret_key" {
  type        = string
  description = "Secret key part for the service account"
  sensitive   = true
}



variable "yc_sa_id" {
  type        = string
  description = "Base Service Account ID"
  sensitive   = true
}
