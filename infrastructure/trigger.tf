locals{
	html_path = "../src/trigger.html"
}
resource "local_file" "html" {
	content = templatefile( local.html_path, {
		api-gateway-url = yandex_api_gateway.gateway.domain	
	})
	filename = "${path.module}/trigger.html"
}
