# Archaeological fragments detection

Small Python application for archaeological fragments detection using [**OpenCV**](https://opencv.org/) and [**YOLO**](https://www.ultralytics.com/). The application executes on [**Yandex Cloud**](https://yandex.cloud/en) infrastructure. It processes images in Yandex Disk folder using [**Yandex Disk REST API**](https://yandex.ru/dev/disk-api/doc/en/). The lifecycle of all infrastructure components is declared and managed deterministically via [**Terraform**](https://developer.hashicorp.com/terraform) and [**Yandex Cloud CLI**](https://yandex.cloud/en/docs/cli/). User can interact with via simple HTML frontend produced by Terraform but oauth_token (credentials to access Yandex Disk) should be provided to Terraform. 


## Automated Infrastructure Deployment

To initialize and deploy the full serverless stack to Yandex Cloud, run the following commands:

### 1. Clone repository to local machine using Git LFS
```bash
git clone https://github.com/ramses55/arch.git
cd arch
```


### 2. Authenticate with the [Yandex Cloud CLI](https://yandex.cloud/en/docs/cli/quickstart)
```bash
yc init
```

### 3. Get credentials for Terraform

  #### a) Navigate to infrastructure/ in cloned repo
     
```bash
cd infrastructure
```

  #### b) Export essential variables

```bash
export YC_TOKEN=$(yc iam create-token) 
export YC_CLOUD_ID=$(yc config get cloud-id)                                                         
export YC_FOLDER_ID=$(yc config get folder-id)
export TF_VAR_folder_id=$(yc config get folder-id)
```
  #### c) Create a service account for terraform. Be sure to save access key pair as the secret value will be shown once.
  

```bash
yc iam service-account create --name admin-sa
yc resource-manager folder add-access-binding $YC_FOLDER_ID --role admin --service-account-name admin-sa
yc iam access-key create --service-account-name admin-sa
```

  #### d) Obtain the oauth_token by following the link
```bash
https://oauth.yandex.ru/authorize?response_type=token&client_id=abd02f50dcb04dbcb7b867e3a1672e7f
```

  #### e) Create a file named `terraform.tfvars` and populate it with parameters:

```hcl
# terraform.tfvars

yc_sa_id      = "<admin-sa ID>" # The unique ID of the service account
yc_access_key = "<ked_id>"      # The AWS-compatible Static Access Key ID
yc_secret_key = "<secret>"      # The corresponding Secret Access Key generated alongside the access key
oauth_token   = "<oauth_token>" # Token for Yandex Disk REST API
```

### 5. Build base infrastructure

```bash
terrafrom init
terrafrom apply
export TAG="cr.yandex/$(terraform output -raw container_registry_url)/worker:latest"
```
### 6. Use registry URL provided by terraform output to build and push docker image

```bash
cd ../src
docker build -t $TAG -f Dockerfile.worker .
docker push $TAG
```
### 7. Update terraform configuration to use proper container image

```bash
cd ../infrastructure
terraform apply -var="container_image=$TAG"
```

### 8. Terraform produced front.html file which is the basic frontend to trigger image processing

